use super::timeline::Timeline;
use crate::TaskGraph;

#[derive(Debug)]
pub(crate) struct Schedule {
    // Batches created by kahns algorithm
    pub(crate) batches: Vec<Vec<usize>>,
    // Node to batch index
    pub(crate) task_to_batch: Vec<usize>,
    // (timeline_index, access_group) each batch touches
    extents: Vec<Vec<(usize, usize)>>,
}

/// Where one access group runs: its span over the waves, from the first
/// member's wave to the last member's (a group can sit on several queues in
/// one wave; the span covers all of them).
#[derive(Clone, Copy)]
pub(crate) struct Placement {
    first: usize,
    last: usize,
}

impl Placement {
    pub(crate) fn start(&self) -> usize { self.first }
    pub(crate) fn end(&self) -> usize { self.last }
}

impl Schedule {
    /// A group's wave span.
    pub(crate) fn placement(&self, timeline: usize, group: usize) -> Placement {
        let (first, last) = self.extents[timeline][group];
        Placement {
            first,
            last,
        }
    }
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    /// Assigns waves and per-group spans (stage 1 of compilation).
    pub(crate) fn schedule(&self, timelines: &[Timeline]) -> Schedule {
        let (batches, task_to_batch) = self.assign_batches(timelines);

        let extents = timelines
            .iter()
            .map(|tl| {
                tl.groups
                    .iter()
                    .map(|group| {
                        let mut first = usize::MAX;
                        let mut last = 0;
                        for &m in &group.members {
                            let b = task_to_batch[m];
                            first = first.min(b);
                            last = last.max(b);
                        }
                        debug_assert!(first != usize::MAX, "group has members");
                        (first, last)
                    })
                    .collect()
            })
            .collect();

        Schedule {
            batches,
            task_to_batch,
            extents,
        }
    }

    pub(crate) fn assign_batches(&self, timelines: &[Timeline]) -> (Vec<Vec<usize>>, Vec<usize>) {
        let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); self.entries.len()];
        for tl in timelines {
            for w in tl.groups.windows(2) {
                for &curr_member in &w[1].members {
                    for &prev_member in &w[0].members {
                        adj_list[curr_member].push(prev_member);
                    }
                }
            }
        }

        let batches = topological_batches(&adj_list);
        let mut task_to_batch = vec![0usize; self.entries.len()];
        for (i, batch) in batches.iter().enumerate() {
            for &t in batch {
                task_to_batch[t] = i;
            }
        }

        (batches, task_to_batch)
    }
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
