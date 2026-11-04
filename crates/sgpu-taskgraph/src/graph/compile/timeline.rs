use std::collections::HashMap;

use sgpu::{AccessType, PipelineStage};

use crate::{PassBufferId, PassImageId, graph::ResourcePool};

use super::{GraphError, NodeDeclaration};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Resource {
    Image(PassImageId),
    Buffer(PassBufferId),
}

/// Where one access group runs on a single queue.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GroupSpan {
    pub(crate) first_batch: usize,
    pub(crate) last_batch: usize,
}

/// A queue-local slice of the schedule
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BatchIndex {
    pub(crate) batch: usize,
    pub(crate) queue: usize,
}

/// The contiguous run of accesses to one resource between two conflict points.
#[derive(Debug)]
pub(crate) struct AccessGroup {
    pub(crate) accesses: Vec<AccessType>,
    /// Node indices, indexed by queue.
    pub(crate) members: [Vec<usize>; 3],
    /// Where this group runs. Empty until the waves are known.
    pub(crate) spans: [Option<GroupSpan>; 3],
}

impl AccessGroup {
    /// Whether this group is exactly one node presenting this resource.
    pub(crate) fn is_present(&self) -> bool { self.accesses.len() == 1 && self.accesses[0] == AccessType::Present && self.member_count() == 1 }

    /// Total nodes across all queues.
    pub(crate) fn member_count(&self) -> usize { self.members.iter().map(Vec::len).sum() }

    /// Assigns Group spans
    pub(crate) fn assign_group_span(&mut self, task_to_batch: &[usize]) {
        for (queue, queue_members) in self.members.iter().enumerate() {
            for &member in queue_members {
                let wave = task_to_batch[member];

                self.spans[queue] = Some(self.spans[queue].map_or(
                    GroupSpan {
                        first_batch: wave,
                        last_batch: wave,
                    },
                    |span| GroupSpan {
                        first_batch: span.first_batch.min(wave),
                        last_batch: span.last_batch.max(wave),
                    },
                ));
            }
        }

        debug_assert!(self.spans.iter().any(Option::is_some), "group has members");
    }

    /// The queues this group runs on.
    pub(crate) fn queues(&self) -> impl Iterator<Item = usize> + '_ { (0..3).filter(move |&queue| self.spans[queue].is_some()) }

    /// Where the group starts on one queue it runs on.
    pub(crate) fn first(&self, queue: usize) -> BatchIndex {
        BatchIndex {
            batch: self.span_on(queue).first_batch,
            queue,
        }
    }

    /// Where the group ends on one queue it runs on.
    pub(crate) fn last(&self, queue: usize) -> BatchIndex {
        BatchIndex {
            batch: self.span_on(queue).last_batch,
            queue,
        }
    }

    fn span_on(&self, queue: usize) -> GroupSpan { self.spans[queue].expect("group does not run on this queue") }

    /// The queue-local start of this group: the earliest `(wave, queue)` it
    /// runs at. Image layout transitions are recorded here, on exactly one
    /// queue; every other queue that touches the group waits on this one.
    pub(crate) fn first_batch(&self) -> BatchIndex { self.queues().map(|queue| self.first(queue)).min().expect("group has members") }

    /// The earliest wave any queue runs this group at.
    pub(crate) fn start(&self) -> usize { self.spans.iter().flatten().map(|span| span.first_batch).min().expect("group has members") }

    /// The latest wave any queue runs this group at.
    pub(crate) fn end(&self) -> usize { self.spans.iter().flatten().map(|span| span.last_batch).max().expect("group has members") }
}

/// Which frame's use of a resource a wait refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Frame {
    /// An earlier use in this frame.
    Current,
    /// The use that ran last frame, on this resource's own slot.
    Previous,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PreviousUse {
    pub(crate) timeline: usize,
    pub(crate) group: usize,
    pub(crate) frame: Frame,
}

#[derive(Debug)]
pub(crate) struct Timeline {
    pub(crate) resource: Resource,
    pub(crate) transient: bool,
    pub(crate) groups: Vec<AccessGroup>,
    /// What must finish before groups[0] starts
    pub(crate) previous_use: PreviousUse,
}

impl Timeline {
    fn push_group(&mut self, access: AccessType, node: usize, queue_index: usize) -> Result<usize, (AccessType, AccessType)> {
        // A check to see whether the last grp contains this node
        if self.groups.last().is_some_and(|last| last.members.iter().any(|members| members.contains(&node))) && conflicts(&self.groups.last().expect("checked above").accesses, &[access]) {
            let last = self.groups.last().expect("checked above");
            return Err((last.accesses.last().copied().unwrap_or(access), access));
        }

        if self.groups.last().is_some_and(|last| !conflicts(&last.accesses, &[access])) {
            let last = self.groups.last_mut().expect("checked above");
            last.accesses.push(access);
            if !last.members[queue_index].contains(&node) {
                last.members[queue_index].push(node);
            }
        } else {
            let mut members = [
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ];
            members[queue_index].push(node);
            self.groups.push(AccessGroup {
                accesses: vec![access],
                members,
                spans: [None; 3],
            });
        }

        Ok(self.groups.len() - 1)
    }

    pub(crate) fn assign_group_spans(&mut self, task_to_batch: &[usize]) {
        for group in &mut self.groups {
            group.assign_group_span(task_to_batch);
        }
    }

    pub(crate) fn validate_present(&self) -> Result<(), GraphError> {
        for (index, group) in self.groups.iter().enumerate() {
            if !group.is_present() {
                continue;
            }
            if index != self.groups.len() - 1 {
                return Err(GraphError::PresentAfterAccess {
                    resource: self.resource,
                });
            }
            if index == 0 {
                return Err(GraphError::PresentWithoutWrite {
                    resource: self.resource,
                });
            }
        }
        Ok(())
    }
}

pub(crate) fn build_timelines<'g, R, D>(pool: &ResourcePool, nodes: &[NodeDeclaration<'g, R, D>]) -> Result<Vec<Timeline>, GraphError> {
    let mut timelines: Vec<Timeline> = Vec::new();
    let mut res_to_timeline: HashMap<Resource, usize> = HashMap::new();

    for (node, declaration) in nodes.iter().enumerate() {
        let queue_index = declaration.queue as usize;

        for &(image, access) in &declaration.images {
            let resource = Resource::Image(image);
            let timeline = *res_to_timeline.entry(resource).or_insert_with(|| {
                timelines.push(Timeline {
                    resource,
                    transient: pool.get_image(image).is_transient(),
                    groups: Vec::new(),
                    previous_use: PreviousUse {
                        timeline: usize::MAX,
                        group: usize::MAX,
                        frame: Frame::Previous,
                    },
                });
                timelines.len() - 1
            });

            if let Err((first, second)) = timelines[timeline].push_group(access, node, queue_index) {
                return Err(GraphError::SelfDependency {
                    node,
                    label: declaration.label,
                    resource,
                    first,
                    second,
                });
            }
        }

        for &(buffer, access) in &declaration.buffers {
            let resource = Resource::Buffer(buffer);
            let timeline = *res_to_timeline.entry(resource).or_insert_with(|| {
                timelines.push(Timeline {
                    resource,
                    transient: pool.get_buffer(buffer).is_transient(),
                    groups: Vec::new(),
                    previous_use: PreviousUse {
                        timeline: usize::MAX,
                        group: usize::MAX,
                        frame: Frame::Previous,
                    },
                });
                timelines.len() - 1
            });

            if let Err((first, second)) = timelines[timeline].push_group(access, node, queue_index) {
                return Err(GraphError::SelfDependency {
                    node,
                    label: declaration.label,
                    resource,
                    first,
                    second,
                });
            }
        }
    }

    for (index, timeline) in timelines.iter_mut().enumerate() {
        timeline.validate_present()?;

        // Set this in general for all resources, later on it is modifed for transients
        timeline.previous_use = PreviousUse {
            timeline: index,
            group: timeline.groups.len() - 1,
            frame: Frame::Previous,
        };
    }

    Ok(timelines)
}

pub(super) fn is_present(accesses: &[AccessType]) -> bool { accesses.contains(&AccessType::Present) }

pub(super) fn is_write(accesses: &[AccessType]) -> bool { accesses.iter().any(|access| access.is_write()) }

/// The combined pipeline stage a group's accesses need.
pub(super) fn stage_of(accesses: &[AccessType]) -> PipelineStage { accesses.iter().fold(PipelineStage::NONE, |acc, &access| acc | access.stage()) }

/// Whether two access sets on the same resource need an ordering between
/// them. Conservatively true for any write, any layout change and `Present`.
pub(super) fn conflicts(previous: &[AccessType], next: &[AccessType]) -> bool { is_write(previous) || is_write(next) || previous.first().zip(next.first()).is_some_and(|(previous, next)| previous.layout_differs(*next)) || is_present(previous) || is_present(next) }
