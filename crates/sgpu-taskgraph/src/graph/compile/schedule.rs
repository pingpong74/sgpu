use super::{GraphError, NodeDeclaration, timeline::Timeline};

/// The order the graph will execute in
#[derive(Debug)]
pub(crate) struct Schedule {
    /// Nodes per wave, each sorted by node index.
    pub(crate) batches: Vec<Vec<usize>>,
}

impl Schedule {
    pub(crate) fn build<'g, R, D>(timelines: &mut [Timeline], nodes: &[NodeDeclaration<'g, R, D>]) -> Result<Schedule, GraphError> {
        let (batches, task_to_batch) = assign_batches(timelines, nodes)?;

        for timeline in timelines.iter_mut() {
            timeline.assign_group_spans(&task_to_batch);
        }

        Ok(Schedule { batches })
    }
}

fn assign_batches<'g, R, D>(timelines: &[Timeline], nodes: &[NodeDeclaration<'g, R, D>]) -> Result<(Vec<Vec<usize>>, Vec<usize>), GraphError> {
    let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for timeline in timelines {
        for pair in timeline.groups.windows(2) {
            for &current in pair[1].members.iter().flatten() {
                for &previous in pair[0].members.iter().flatten() {
                    adj_list[current].push(previous);
                }
            }
        }
    }

    let batches = topological_batches(&adj_list, nodes)?;

    let mut task_to_batch = vec![0usize; nodes.len()];
    for (batch_index, batch) in batches.iter().enumerate() {
        for &task in batch {
            task_to_batch[task] = batch_index;
        }
    }

    Ok((batches, task_to_batch))
}

// Kahn's algorithm for topological sort
fn topological_batches<'g, R, D>(adj_list: &[Vec<usize>], nodes: &[NodeDeclaration<'g, R, D>]) -> Result<Vec<Vec<usize>>, GraphError> {
    use std::collections::VecDeque;

    let mut in_degree = vec![0usize; adj_list.len()];
    let mut forward: Vec<Vec<usize>> = vec![Vec::new(); adj_list.len()];
    for (node, predecessors) in adj_list.iter().enumerate() {
        for &previous in predecessors {
            forward[previous].push(node);
            in_degree[node] += 1;
        }
    }

    let mut queue: VecDeque<usize> = (0..adj_list.len()).filter(|&node| in_degree[node] == 0).collect();
    let mut batches: Vec<Vec<usize>> = Vec::new();

    while !queue.is_empty() {
        let mut batch = Vec::with_capacity(queue.len());
        for _ in 0..queue.len() {
            let node = queue.pop_front().expect("queue checked non-empty");
            batch.push(node);
            for &next in &forward[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }
        batch.sort_unstable();
        batches.push(batch);
    }

    let scheduled: usize = batches.iter().map(Vec::len).sum();
    if scheduled != adj_list.len() {
        let mut placed = vec![false; adj_list.len()];
        for batch in &batches {
            for &node in batch {
                placed[node] = true;
            }
        }

        let unplaced: Vec<_> = (0..adj_list.len()).filter(|&node| !placed[node]).collect();
        let labels = unplaced.clone().iter().map(|&i| nodes[i].label).collect();

        return Err(GraphError::DependencyCycle {
            nodes: unplaced,
            labels,
        });
    }

    Ok(batches)
}
