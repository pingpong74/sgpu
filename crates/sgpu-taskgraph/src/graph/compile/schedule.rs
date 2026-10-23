use sgpu::QueueType;

use super::timeline::Timeline;

#[derive(Debug)]
pub(crate) struct Schedule {
    // Batches created by Kahn's algorithm
    pub(crate) batches: Vec<Vec<usize>>,
    // Wave span of every access group
    spans: Vec<Vec<GroupSpan>>,
    // Node to queue index
    node_queue: Vec<usize>,
}

// Where one access group runs: its wave span on each queue.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GroupSpan {
    first_wave: [Option<usize>; 3],
    last_wave: [Option<usize>; 3],
}

// A queue-local slice of the schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SliceId {
    pub(crate) wave: usize,
    pub(crate) queue: usize,
}

impl GroupSpan {
    fn new(members: &[usize], task_to_batch: &[usize], node_queue: &[usize]) -> Self {
        let mut span = GroupSpan {
            first_wave: [None; 3],
            last_wave: [None; 3],
        };
        for &member in members {
            let (batch, queue) = (task_to_batch[member], node_queue[member]);
            span.first_wave[queue] = Some(span.first_wave[queue].map_or(batch, |first| first.min(batch)));
            span.last_wave[queue] = Some(span.last_wave[queue].map_or(batch, |last| last.max(batch)));
        }
        debug_assert!(span.first_wave.iter().any(Option::is_some), "group has members");
        span
    }

    pub(crate) fn queues(&self) -> impl Iterator<Item = usize> + '_ { (0..3).filter(move |&q| self.first_wave[q].is_some()) }

    pub(crate) fn first(&self, queue: usize) -> SliceId {
        SliceId {
            wave: self.first_wave[queue].expect("group does not run on this queue"),
            queue,
        }
    }

    pub(crate) fn last(&self, queue: usize) -> SliceId {
        SliceId {
            wave: self.last_wave[queue].expect("group does not run on this queue"),
            queue,
        }
    }

    pub(crate) fn anchor(&self) -> SliceId { self.queues().map(|queue| self.first(queue)).min().expect("group has members") }

    pub(crate) fn start(&self) -> usize { self.first_wave.iter().flatten().copied().min().expect("group has members") }

    pub(crate) fn end(&self) -> usize { self.last_wave.iter().flatten().copied().max().expect("group has members") }
}

impl Schedule {
    pub(crate) fn build(timelines: &[Timeline], node_queues: &[QueueType]) -> Schedule {
        let (batches, task_to_batch) = assign_batches(timelines, node_queues.len());
        let node_queue: Vec<usize> = node_queues.iter().map(|&queue| queue as usize).collect();
        let spans = timelines.iter().map(|timeline| timeline.groups.iter().map(|group| GroupSpan::new(&group.members, &task_to_batch, &node_queue)).collect()).collect();

        Schedule {
            batches,
            spans,
            node_queue,
        }
    }

    pub(crate) fn span(&self, timeline: usize, group: usize) -> GroupSpan { self.spans[timeline][group] }

    pub(crate) fn queue_of(&self, node: usize) -> usize { self.node_queue[node] }

    pub(crate) fn node_count(&self) -> usize { self.node_queue.len() }
}

fn assign_batches(timelines: &[Timeline], node_count: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    for timeline in timelines {
        for pair in timeline.groups.windows(2) {
            for &current in &pair[1].members {
                for &previous in &pair[0].members {
                    adj_list[current].push(previous);
                }
            }
        }
    }

    let batches = topological_batches(&adj_list);
    let mut task_to_batch = vec![0usize; node_count];
    for (batch_index, batch) in batches.iter().enumerate() {
        for &task in batch {
            task_to_batch[task] = batch_index;
        }
    }

    (batches, task_to_batch)
}

// Kahn's algorithm for topological sort
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
