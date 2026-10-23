use std::collections::{BTreeMap, HashMap};

use sgpu::{Buffer, Image, MemoryType, PooledBufferDescription, PooledImageDescription, PooledResourceDescription, destroy_buffer, destroy_image};

use super::{
    schedule::Schedule,
    timeline::{Resource, Timeline, is_write},
};
use crate::{
    TaskGraph,
    resource::{InnerBuffer, InnerImage},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cycle {
    Same,
    Previous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Predecessor {
    pub(crate) timeline: usize,
    pub(crate) cycle: Cycle,
}

pub(crate) struct AliasMember {
    pub(crate) res_index: u32,
    pub(crate) offset: u64,
    pub(crate) aliased: bool,
}

pub(crate) struct AliasGroup {
    pub(crate) images: Vec<AliasMember>,
    pub(crate) buffers: Vec<AliasMember>,
}

pub(crate) struct AliasSchedule {
    pub(crate) groups: Vec<AliasGroup>,
    predecessors: HashMap<usize, Predecessor>,
}

impl AliasSchedule {
    pub(crate) fn predecessor(&self, timeline: usize) -> Predecessor {
        self.predecessors.get(&timeline).copied().unwrap_or(Predecessor {
            timeline,
            cycle: Cycle::Previous,
        })
    }
}

struct Owner {
    timeline: usize,
    resource: Resource,
}

struct Lane {
    memory_type: MemoryType,
    owners: Vec<Owner>,
}

struct Life {
    timeline: usize,
    resource: Resource,
    start: usize,
    end: usize,
}

impl<'g, R, D> TaskGraph<'g, R, D> {
    pub(crate) fn schedule_aliasing(&self, timelines: &[Timeline], sched: &Schedule) -> AliasSchedule {
        let lanes = self.share(timelines, sched);
        let predecessors = predecessors(&lanes);
        let groups = self.lay_out(&lanes);
        AliasSchedule {
            groups,
            predecessors,
        }
    }

    fn share(&self, timelines: &[Timeline], sched: &Schedule) -> Vec<Lane> {
        let mut classes: BTreeMap<usize, (MemoryType, Vec<Life>)> = BTreeMap::new();
        for (ti, tl) in timelines.iter().enumerate().filter(|(_, tl)| tl.transient) {
            assert!(is_write(&tl.groups[0].accesses), "transient resource {:?} must start with a write", tl.resource);

            let memory_type = self.transient_memory_type(tl.resource);
            classes.entry(memory_type as usize).or_insert_with(|| (memory_type, Vec::new())).1.push(Life {
                timeline: ti,
                resource: tl.resource,
                start: sched.placement(ti, 0).start(),
                end: sched.placement(ti, tl.groups.len() - 1).end(),
            });
        }

        let mut lanes = Vec::new();
        for (_, (memory_type, mut lives)) in classes {
            lives.sort_by_key(|life| (life.start, life.timeline));

            let mut open: Vec<(usize, Vec<Owner>)> = Vec::new();
            for life in lives {
                let owner = Owner {
                    timeline: life.timeline,
                    resource: life.resource,
                };
                match open.iter_mut().find(|(end, _)| *end < life.start) {
                    Some((end, owners)) => {
                        *end = life.end;
                        owners.push(owner);
                    }
                    None => open.push((life.end, vec![owner])),
                }
            }
            lanes.extend(open.into_iter().map(|(_, owners)| Lane {
                memory_type,
                owners,
            }));
        }
        lanes
    }

    fn lay_out(&self, lanes: &[Lane]) -> Vec<AliasGroup> {
        let mut classes: BTreeMap<usize, Vec<&Lane>> = BTreeMap::new();
        for lane in lanes {
            classes.entry(lane.memory_type as usize).or_default().push(lane);
        }

        let mut groups = Vec::with_capacity(classes.len());
        for (_, lanes) in classes {
            let mut watermark = 0u64;
            let mut images = Vec::new();
            let mut buffers = Vec::new();

            for lane in lanes {
                let reqs: Vec<sgpu::ResourceRequirements> = lane.owners.iter().map(|owner| self.requirements(owner.resource)).collect();
                let size = reqs.iter().map(|r| r.size).max().expect("lane has owners");
                let lane_alignment = reqs.iter().map(|r| r.alignment).max().expect("lane has owners");

                let offset = watermark.div_ceil(lane_alignment) * lane_alignment;
                watermark = offset + size;

                let image_owners = lane.owners.iter().filter(|owner| matches!(owner.resource, Resource::Image(_))).count();
                for owner in &lane.owners {
                    match owner.resource {
                        Resource::Image(i) => images.push(AliasMember {
                            res_index: i,
                            offset,
                            aliased: image_owners >= 2,
                        }),
                        Resource::Buffer(b) => buffers.push(AliasMember {
                            res_index: b,
                            offset,
                            aliased: false,
                        }),
                    }
                }
            }

            groups.push(AliasGroup {
                images,
                buffers,
            });
        }
        groups
    }

    fn transient_memory_type(&self, resource: Resource) -> MemoryType {
        match resource {
            Resource::Image(i) => {
                let InnerImage::Transient {
                    image_desc, ..
                } = &self.images[i as usize]
                else {
                    unreachable!("timeline marked transient")
                };
                image_desc.memory_type
            }
            Resource::Buffer(b) => {
                let InnerBuffer::Transient {
                    buffer_desc,
                    ..
                } = &self.buffers[b as usize]
                else {
                    unreachable!("timeline marked transient")
                };
                buffer_desc.memory_type
            }
        }
    }

    fn requirements(&self, resource: Resource) -> sgpu::ResourceRequirements {
        match resource {
            Resource::Image(i) => {
                let InnerImage::Transient {
                    image_desc, ..
                } = &self.images[i as usize]
                else {
                    unreachable!("lane owner is transient")
                };
                sgpu::image_requirements(image_desc)
            }
            Resource::Buffer(b) => {
                let InnerBuffer::Transient {
                    buffer_desc,
                    ..
                } = &self.buffers[b as usize]
                else {
                    unreachable!("lane owner is transient")
                };
                sgpu::buffer_requirements(buffer_desc)
            }
        }
    }

    /// Rebuild all transient instances from the alias layout.
    /// The caller must have waited for the GPU before calling.
    pub(crate) fn materialize_transients(&mut self, groups: &[AliasGroup]) {
        let slots = self.frames_in_flight;

        let old_images: Vec<(usize, Vec<Image>)> = self
            .images
            .iter_mut()
            .enumerate()
            .filter_map(|(index, rack)| match rack {
                InnerImage::Transient { images, .. } => Some((index, std::mem::take(images))),
                _ => None,
            })
            .collect();
        let old_buffers: Vec<Vec<Buffer>> = self
            .buffers
            .iter_mut()
            .filter_map(|rack| match rack {
                InnerBuffer::Transient {
                    buffers, ..
                } => Some(std::mem::take(buffers)),
                _ => None,
            })
            .collect();

        for (index, images) in old_images {
            if images.is_empty() {
                continue;
            }
            self.views.destroy_slot(index as u32);
            for image in images {
                destroy_image(image);
            }
        }
        for buffers in old_buffers {
            for buffer in buffers {
                destroy_buffer(buffer);
            }
        }

        for group in groups {
            let mut image_inputs: Vec<PooledImageDescription> = Vec::with_capacity(group.images.len());
            let mut image_indices: Vec<u32> = Vec::with_capacity(group.images.len());
            for member in &group.images {
                let InnerImage::Transient {
                    image_desc, ..
                } = &self.images[member.res_index as usize]
                else {
                    unreachable!("alias group member is transient");
                };
                image_inputs.push(PooledImageDescription {
                    desc: *image_desc,
                    offset: member.offset,
                    aliased: member.aliased,
                });
                image_indices.push(member.res_index);
            }

            let mut buffer_inputs: Vec<PooledBufferDescription> = Vec::with_capacity(group.buffers.len());
            let mut buffer_indices: Vec<u32> = Vec::with_capacity(group.buffers.len());
            for member in &group.buffers {
                let InnerBuffer::Transient {
                    buffer_desc,
                    ..
                } = &self.buffers[member.res_index as usize]
                else {
                    unreachable!("alias group member is transient");
                };
                buffer_inputs.push(PooledBufferDescription {
                    desc: *buffer_desc,
                    offset: member.offset,
                });
                buffer_indices.push(member.res_index);
            }

            for slot in 0..slots {
                // SAFETY: lane offsets are non-overlapping unless their
                // lifetimes are ordered by the alias predecessor edges.
                let (slot_images, slot_buffers) = unsafe {
                    sgpu::create_pooled_resources(&PooledResourceDescription {
                        images: &image_inputs,
                        buffers: &buffer_inputs,
                    })
                };

                for (member_index, rack) in image_indices.iter().enumerate() {
                    let InnerImage::Transient { images, .. } = &mut self.images[*rack as usize] else {
                        unreachable!("transient member above")
                    };
                    debug_assert_eq!(images.len(), slot, "rack {rack} instance sets drifted between slots");
                    images.push(slot_images[member_index]);
                }
                for (member_index, rack) in buffer_indices.iter().enumerate() {
                    let InnerBuffer::Transient {
                        buffers, ..
                    } = &mut self.buffers[*rack as usize]
                    else {
                        unreachable!("transient member above")
                    };
                    debug_assert_eq!(buffers.len(), slot, "rack {rack} instance sets drifted between slots");
                    buffers.push(slot_buffers[member_index]);
                }
            }

            debug_assert!(image_indices.iter().all(|&rack| { matches!(&self.images[rack as usize], InnerImage::Transient { images, .. } if images.len() == slots) }));
            debug_assert!(buffer_indices.iter().all(|&rack| { matches!(&self.buffers[rack as usize], InnerBuffer::Transient { buffers, .. } if buffers.len() == slots) }));
        }
    }
}

fn predecessors(lanes: &[Lane]) -> HashMap<usize, Predecessor> {
    let mut predecessors = HashMap::new();
    for lane in lanes.iter().filter(|lane| lane.owners.len() >= 2) {
        for pair in lane.owners.windows(2) {
            predecessors.insert(
                pair[1].timeline,
                Predecessor {
                    timeline: pair[0].timeline,
                    cycle: Cycle::Same,
                },
            );
        }
        predecessors.insert(
            lane.owners[0].timeline,
            Predecessor {
                timeline: lane.owners[lane.owners.len() - 1].timeline,
                cycle: Cycle::Previous,
            },
        );
    }
    predecessors
}
