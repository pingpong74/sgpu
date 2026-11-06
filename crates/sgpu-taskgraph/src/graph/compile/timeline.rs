use std::collections::HashMap;

use sgpu::{AccessType, ImageAspect, ImageSubresources, PipelineStage};

use crate::{PassBufferId, PassImageId, graph::ResourcePool, resource::ImageViewRef};

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

/// One access inside a group, together with the view range it touches.
#[derive(Clone, Debug, Copy)]
pub(crate) struct GroupAccess {
    pub(crate) access: AccessType,
    pub(crate) view: ImageViewRef,
    /// The subresources this access touches. `None` means the whole image
    /// (its default view) or a buffer, and overlaps every other range.
    pub(crate) range: Option<ImageSubresources>,
}

/// The contiguous run of accesses to one resource between two conflict points.
#[derive(Debug)]
pub(crate) struct AccessGroup {
    pub(crate) accesses: Vec<GroupAccess>,
    /// Node indices, indexed by queue.
    pub(crate) members: [Vec<usize>; 3],
    /// Where this group runs. Empty until the waves are known.
    pub(crate) spans: [Option<GroupSpan>; 3],
}

impl AccessGroup {
    pub(crate) fn is_write(&self) -> bool { self.accesses.iter().any(|access| access.access.is_write()) }

    pub(crate) fn has_present(&self) -> bool { self.accesses.iter().any(|access| access.access == AccessType::Present) }

    pub(crate) fn stage(&self) -> PipelineStage { self.accesses.iter().fold(PipelineStage::NONE, |acc, access| acc | access.access.stage()) }

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

    /// Earlier (queue, batch)
    pub(crate) fn first_batch(&self) -> BatchIndex { self.queues().map(|queue| self.first(queue)).min().expect("group has members") }

    /// The earliest batch any queue runs this group at.
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
    fn push_group(&mut self, access: GroupAccess, node: usize, queue_index: usize) -> Result<usize, (AccessType, AccessType)> {
        if let Some(last) = self.groups.last()
            && last.members.iter().any(|members| members.contains(&node))
            && let Some(conflict) = first_conflict(&last.accesses, &[access])
        {
            return Err(conflict);
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
            if !group.has_present() {
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

        for &img_use in &declaration.images {
            let resource = Resource::Image(img_use.image);
            let timeline = *res_to_timeline.entry(resource).or_insert_with(|| {
                timelines.push(Timeline {
                    resource,
                    transient: pool.get_image(img_use.image).is_transient(),
                    groups: Vec::new(),
                    previous_use: PreviousUse {
                        timeline: usize::MAX,
                        group: usize::MAX,
                        frame: Frame::Previous,
                    },
                });
                timelines.len() - 1
            });

            if let Err((first, second)) = timelines[timeline].push_group(
                GroupAccess {
                    access: img_use.access,
                    view: img_use.view,
                    range: match img_use.view {
                        ImageViewRef::Default => None,
                        ImageViewRef::Registered(view) => Some(pool.view_subresources(view)),
                    },
                },
                node,
                queue_index,
            ) {
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

            if let Err((first, second)) = timelines[timeline].push_group(
                GroupAccess {
                    access,
                    view: ImageViewRef::Default,
                    range: None,
                },
                node,
                queue_index,
            ) {
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

pub(super) fn conflicts(previous: &[GroupAccess], next: &[GroupAccess]) -> bool { first_conflict(previous, next).is_some() }

pub(super) fn first_conflict(previous: &[GroupAccess], next: &[GroupAccess]) -> Option<(AccessType, AccessType)> {
    previous.iter().find_map(|previous| next.iter().find(|next| ranges_overlap(previous.range, next.range) && access_conflicts(previous.access, next.access)).map(|next| (previous.access, next.access)))
}

pub(super) fn access_conflicts(previous: AccessType, next: AccessType) -> bool { previous.is_write() || next.is_write() || previous.layout_differs(next) || previous == AccessType::Present || next == AccessType::Present }

/// Whether two subresource ranges overlap.
pub(super) fn ranges_overlap(previous: Option<ImageSubresources>, next: Option<ImageSubresources>) -> bool {
    let (Some(previous), Some(next)) = (previous, next) else {
        return true;
    };

    let levels = previous.mip_level < span_end(next.mip_level, next.level_count) && next.mip_level < span_end(previous.mip_level, previous.level_count);
    let layers = previous.base_array_layer < span_end(next.base_array_layer, next.layer_count) && next.base_array_layer < span_end(previous.base_array_layer, previous.layer_count);

    // `AUTO` is resolved against the image's format when the view is created,
    // which the plan cannot see, so it overlaps every aspect.
    let aspects = previous.aspect.intersects(next.aspect) || previous.aspect == ImageAspect::AUTO || next.aspect == ImageAspect::AUTO;

    aspects && levels && layers
}

/// Whether inner is inside outer
pub(super) fn range_contains(outer: Option<ImageSubresources>, inner: Option<ImageSubresources>) -> bool {
    let (Some(outer), Some(inner)) = (outer, inner) else {
        return false;
    };

    // A count of `0` means "rest of the image".
    let levels = outer.mip_level <= inner.mip_level && span_end(inner.mip_level, inner.level_count) <= span_end(outer.mip_level, outer.level_count);
    let layers = outer.base_array_layer <= inner.base_array_layer && span_end(inner.base_array_layer, inner.layer_count) <= span_end(outer.base_array_layer, outer.layer_count);

    outer.aspect.contains(inner.aspect) && levels && layers
}

fn span_end(base: u32, count: u32) -> u32 {
    if count == 0 {
        u32::MAX
    } else {
        base.saturating_add(count)
    }
}
