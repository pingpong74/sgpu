use sgpu::{AccessType, PipelineStage, QueueType};

use super::timeline::{Resource, Timeline, conflicts, is_write, stage_of};
use super::{ErasedNode, TaskGraph};

const QUEUES: [QueueType; 3] = [
    QueueType::Graphics,
    QueueType::Compute,
    QueueType::Transfer,
];

// Node + the resources it accesses
pub(super) struct NodeDeclaration<'g, R, D> {
    pub(super) built: Box<dyn ErasedNode<R, D> + 'g>,
    pub(super) images: Vec<(u32, AccessType)>,
    pub(super) buffers: Vec<(u32, AccessType)>,
}

#[derive(Clone)]
pub(super) struct CompiledImageBarrier {
    pub(super) image: u32,
    pub(super) previous: Vec<AccessType>,
    pub(super) next: Vec<AccessType>,
    pub(super) discard: bool,
}

#[derive(Default)]
pub(super) struct CompiledGlobalBarrier {
    pub(super) previous: Vec<AccessType>,
    pub(super) next: Vec<AccessType>,
}

#[derive(Default)]
pub(super) struct CompiledBarriers {
    pub(super) images: Vec<CompiledImageBarrier>,
    pub(super) global: CompiledGlobalBarrier,
}

// One queue's share of one batch
pub(super) struct QueueBatch {
    pub(super) queue: QueueType,
    pub(super) nodes: Vec<usize>,
    // This is counter of queues from this frame
    pub(super) waits: Vec<(QueueType, PipelineStage)>,
    // This is counters of queues from previous frames
    pub(super) prev_frame_waits: Vec<(QueueType, PipelineStage)>,
    pub(super) acquire: bool,
    // resources start undefined.
    pub(super) undefined_barriers: CompiledBarriers,
    // first accesses ordered after the previous frame's final ones.
    pub(super) barriers: CompiledBarriers,
}

impl QueueBatch {
    fn new(queue: QueueType, nodes: Vec<usize>) -> QueueBatch {
        QueueBatch {
            queue,
            nodes,
            waits: Vec::new(),
            prev_frame_waits: Vec::new(),
            acquire: false,
            undefined_barriers: CompiledBarriers::default(),
            barriers: CompiledBarriers::default(),
        }
    }
}

pub(super) struct CompiledGraph<'g, R, D> {
    pub(super) batches: Vec<Vec<QueueBatch>>,
    pub(super) built: Vec<Box<dyn ErasedNode<R, D> + 'g>>,
    pub(super) present: bool,
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    pub(super) fn adj_list(&self, timelines: &[Timeline], present: Option<usize>) -> Vec<Vec<usize>> {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); self.entries.len()];
        for (ti, tl) in timelines.iter().enumerate() {
            for pair in tl.groups.windows(2) {
                for &dm in &pair[1].members {
                    for &sm in &pair[0].members {
                        adj[dm].push(sm);
                    }
                }
            }

            if Some(ti) == present {
                let first = &tl.groups[0];
                for &m in &first.members[1..] {
                    adj[m].push(first.members[0]);
                }
            }
        }
        adj
    }

    pub(super) fn assign_batches(&self, timelines: &mut [Timeline], adj: &[Vec<usize>], present: Option<usize>) -> (Vec<Vec<usize>>, Vec<usize>) {
        let batches = topological_batches(adj);

        let mut task_to_batch = vec![0usize; self.entries.len()];
        for (i, batch) in batches.iter().enumerate() {
            for &t in batch {
                task_to_batch[t] = i;
            }
        }

        for tl in timelines.iter_mut() {
            for g in &mut tl.groups {
                g.batch = g.members.iter().map(|&m| task_to_batch[m]).max().unwrap_or(0);
            }
        }

        if let Some(ti) = present {
            timelines[ti].groups.last_mut().unwrap().batch = batches.len();
        }

        (batches, task_to_batch)
    }

    pub(super) fn queue_batches(&self, timelines: &[Timeline], batches: &[Vec<usize>], task_to_batch: &[usize], present: Option<usize>) -> Vec<Vec<QueueBatch>> {
        let mut queue_batches: Vec<Vec<QueueBatch>> = batches
            .iter()
            .map(|batch| {
                QUEUES
                    .iter()
                    .filter_map(|&q| {
                        let nodes: Vec<usize> = batch.iter().copied().filter(|&n| self.entries[n].queue == q).collect();
                        (!nodes.is_empty()).then(|| QueueBatch::new(q, nodes))
                    })
                    .collect()
            })
            .collect();

        if let Some(ti) = present {
            let tl = &timelines[ti];
            let queue = tl.groups.last().unwrap().queue;
            let mut present_batch = QueueBatch::new(queue, Vec::new());

            let real_last = &tl.groups[tl.groups.len() - 2];
            for &m in &real_last.members {
                let mq = self.entries[m].queue;
                if mq != queue {
                    add_wait(&mut present_batch, mq, stage_of(&real_last.accesses));
                }
            }

            let first = &tl.groups[0];
            let anchor = first.members[0];
            let anchor_queue = self.entries[anchor].queue;
            queue_batch_mut(&mut queue_batches, task_to_batch[anchor], anchor_queue).acquire = true;
            for &m in &first.members[1..] {
                let mq = self.entries[m].queue;
                if mq != anchor_queue {
                    add_wait(queue_batch_mut(&mut queue_batches, task_to_batch[m], mq), anchor_queue, stage_of(&first.accesses));
                }
            }

            queue_batches.push(vec![present_batch]);
        }

        queue_batches
    }

    pub(super) fn derive_sync(&self, timelines: &[Timeline], task_to_batch: &[usize], queue_batches: &mut [Vec<QueueBatch>]) {
        for tl in timelines {
            for pair in tl.groups.windows(2) {
                let (src, dst) = (&pair[0], &pair[1]);
                if !conflicts(&src.accesses, &dst.accesses) {
                    continue;
                }

                let dst_stage = stage_of(&dst.accesses);
                for &dm in &dst.members {
                    for &sm in &src.members {
                        let (dq, sq) = (self.entries[dm].queue, self.entries[sm].queue);
                        if dq != sq {
                            add_wait(queue_batch_mut(queue_batches, task_to_batch[dm], dq), sq, dst_stage);
                        }
                    }
                }

                let queue_batch = queue_batch_mut(queue_batches, dst.batch, dst.queue);
                match tl.resource {
                    Resource::Image(img) => {
                        let spec = CompiledImageBarrier {
                            image: img,
                            previous: src.accesses.clone(),
                            next: dst.accesses.clone(),
                            discard: false,
                        };
                        queue_batch.undefined_barriers.images.push(spec.clone());
                        queue_batch.barriers.images.push(spec);
                    }
                    Resource::Buffer(_) => {
                        for barriers in [
                            &mut queue_batch.undefined_barriers,
                            &mut queue_batch.barriers,
                        ] {
                            barriers.global.previous.extend_from_slice(&src.accesses);
                            barriers.global.next.extend_from_slice(&dst.accesses);
                        }
                    }
                }
            }

            let first = &tl.groups[0];
            let last = &tl.groups[tl.groups.len() - 1];
            let conflict = conflicts(&first.accesses, &last.accesses);

            if conflict && !last.present {
                for &fm in &first.members {
                    for &lm in &last.members {
                        let (fq, lq) = (self.entries[fm].queue, self.entries[lm].queue);
                        if fq != lq {
                            add_prev_frame_wait(queue_batch_mut(queue_batches, task_to_batch[fm], fq), lq, stage_of(&first.accesses));
                        }
                    }
                }
            }

            let anchor = first.members[0];
            let (fb, fq) = (task_to_batch[anchor], self.entries[anchor].queue);

            match tl.resource {
                Resource::Image(img) => {
                    if is_write(&first.accesses) {
                        queue_batch_mut(queue_batches, fb, fq).undefined_barriers.images.push(CompiledImageBarrier {
                            image: img,
                            previous: vec![AccessType::None],
                            next: first.accesses.clone(),
                            discard: true,
                        });
                    }

                    if conflict {
                        queue_batch_mut(queue_batches, fb, fq).barriers.images.push(CompiledImageBarrier {
                            image: img,
                            previous: last.accesses.clone(),
                            next: first.accesses.clone(),
                            discard: last.present && is_write(&first.accesses),
                        });
                    }
                }
                Resource::Buffer(_) => {
                    if conflict {
                        let queue_batch = queue_batch_mut(queue_batches, fb, fq);
                        queue_batch.barriers.global.previous.extend_from_slice(&last.accesses);
                        queue_batch.barriers.global.next.extend_from_slice(&first.accesses);
                    }
                }
            }
        }
    }
}

fn queue_batch_mut(queue_batches: &mut [Vec<QueueBatch>], batch: usize, queue: QueueType) -> &mut QueueBatch { queue_batches[batch].iter_mut().find(|b| b.queue == queue).expect("batch has a queue batch for this queue") }

fn add_wait(queue_batch: &mut QueueBatch, src: QueueType, stage: PipelineStage) { push_wait(&mut queue_batch.waits, src, stage) }

fn add_prev_frame_wait(queue_batch: &mut QueueBatch, src: QueueType, stage: PipelineStage) { push_wait(&mut queue_batch.prev_frame_waits, src, stage) }

fn push_wait(waits: &mut Vec<(QueueType, PipelineStage)>, src: QueueType, stage: PipelineStage) {
    match waits.iter_mut().find(|(q, _)| *q == src) {
        Some(existing) => existing.1 |= stage,
        None => waits.push((src, stage)),
    }
}

/// Kahn's algorithm for topological sort
pub(super) fn topological_batches(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    use std::collections::VecDeque;

    let mut in_degree = vec![0usize; adj.len()];
    let mut forward = vec![Vec::new(); adj.len()];
    for (u, predecessors) in adj.iter().enumerate() {
        for &v in predecessors {
            forward[v].push(u);
            in_degree[u] += 1;
        }
    }

    let mut queue: VecDeque<usize> = (0..adj.len()).filter(|&i| in_degree[i] == 0).collect();
    let mut batches: Vec<Vec<usize>> = Vec::new();

    while !queue.is_empty() {
        let mut batch = Vec::with_capacity(queue.len());
        for _ in 0..queue.len() {
            let u = queue.pop_front().unwrap();
            batch.push(u);
            for &v in &forward[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
        batch.sort_unstable();
        batches.push(batch);
    }

    assert_eq!(batches.iter().map(Vec::len).sum::<usize>(), adj.len(), "internal dependency cycle");
    batches
}
