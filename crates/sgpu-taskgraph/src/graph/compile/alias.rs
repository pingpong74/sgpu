use std::collections::BTreeMap;

use crate::{
    PassBufferId, PassImageId,
    graph::{InnerBuffer, InnerImage, ResourcePool},
};

use super::{
    GraphError,
    timeline::{Frame, PreviousUse, Resource, Timeline},
};

/// All lanes of one memory type, packed into a single allocation.
pub(crate) struct Heap {
    pub(crate) alias_grps: Vec<AliasGroup>,
}

/// Resources whose lifetimes never overlap
pub(crate) struct AliasGroup {
    /// Two or more images share this group, so they must be created aliased.
    pub(crate) aliased: bool,
    pub(crate) members: Vec<HeapMember>,
}

pub(crate) enum HeapMember {
    Image(PassImageId),
    Buffer(PassBufferId),
}

impl HeapMember {
    /// The graph resource this heap slot is for.
    pub(crate) fn resource(&self) -> Resource {
        match self {
            HeapMember::Image(id) => Resource::Image(*id),
            HeapMember::Buffer(id) => Resource::Buffer(*id),
        }
    }
}

/// One transient resource's span of life, measured in schedule waves.
struct Life {
    timeline: usize,
    member: HeapMember,
    start: usize,
    end: usize,
}

/// Decides which transients share memory.
pub(crate) fn build_alias_plan(pool: &ResourcePool, timelines: &[Timeline]) -> Result<Vec<Heap>, GraphError> {
    let classes = collect_lives(pool, timelines)?;

    let mut heaps = Vec::with_capacity(classes.len());

    for (_, mut lives) in classes {
        lives.sort_by_key(|life| (life.start, life.timeline));

        // Greedily create the lanes
        let lanes = {
            let mut lanes: Vec<Vec<Life>> = Vec::new();

            for life in lives {
                match lanes.iter_mut().find(|lane| lane.last().is_some_and(|last| last.end < life.start)) {
                    Some(lane) => lane.push(life),
                    None => lanes.push(vec![life]),
                }
            }

            lanes
        };

        heaps.push(Heap {
            alias_grps: lanes
                .into_iter()
                .map(|lives| {
                    let image_count = lives.iter().filter(|life| matches!(life.member, HeapMember::Image(_))).count();

                    AliasGroup {
                        aliased: image_count >= 2,
                        members: lives.into_iter().map(|life| life.member).collect(),
                    }
                })
                .collect(),
        });
    }

    Ok(heaps)
}

/// Groups every transient's lifetime by memory type, in ascending type order.
fn collect_lives(pool: &ResourcePool, timelines: &[Timeline]) -> Result<BTreeMap<usize, Vec<Life>>, GraphError> {
    let mut classes: BTreeMap<usize, Vec<Life>> = BTreeMap::new();

    for (timeline_index, timeline) in timelines.iter().enumerate().filter(|(_, tl)| tl.transient) {
        if !timeline.groups[0].is_write() {
            return Err(GraphError::TransientNotWrittenFirst {
                resource: timeline.resource,
            });
        }

        let (member, memory_type) = match timeline.resource {
            Resource::Image(index) => {
                let InnerImage::Transient {
                    image_desc, ..
                } = pool.get_image(index)
                else {
                    unreachable!("timeline marked transient")
                };
                (HeapMember::Image(index), image_desc.memory_type as usize)
            }
            Resource::Buffer(index) => {
                let InnerBuffer::Transient {
                    buffer_desc,
                    ..
                } = pool.get_buffer(index)
                else {
                    unreachable!("timeline marked transient")
                };
                (HeapMember::Buffer(index), buffer_desc.memory_type as usize)
            }
        };

        classes.entry(memory_type).or_default().push(Life {
            timeline: timeline_index,
            member,
            start: timeline.groups[0].start(),
            end: timeline.groups[timeline.groups.len() - 1].end(),
        });
    }

    Ok(classes)
}

/// Writes each transient's lane-derived `previous_use`.
pub(crate) fn assign_lane_waits(heaps: &[Heap], timelines: &mut [Timeline]) {
    for heap in heaps {
        for lane in &heap.alias_grps {
            for (position, member) in lane.members.iter().enumerate() {
                let previous = if position == 0 {
                    lane.members.last().expect("lane has owners")
                } else {
                    &lane.members[position - 1]
                };

                let timeline = timelines.iter().position(|timeline| timeline.resource == member.resource()).expect("lane member has a timeline");
                let previous_timeline = timelines.iter().position(|timeline| timeline.resource == previous.resource()).expect("lane member has a timeline");

                timelines[timeline].previous_use = PreviousUse {
                    timeline: previous_timeline,
                    group: timelines[previous_timeline].groups.len() - 1,
                    frame: if position == 0 {
                        Frame::Previous
                    } else {
                        Frame::Current
                    },
                };
            }
        }
    }
}
