use std::collections::BTreeMap;

use sgpu::{Buffer, BufferDescription, Image, ImageDescription, MemoryType, PooledBufferDescription, PooledImageDescription, PooledResourceDescription, ResourceRequirements, destroy_buffer, destroy_image};

use super::{
    schedule::Schedule,
    timeline::{Resource, Timeline, is_write},
};
use crate::{
    TaskGraph,
    resource::{InnerBuffer, InnerImage},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Frame {
    Current,
    Previous,
}

// The use that must finish before a timeline's first group starts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreviousUse {
    pub(crate) timeline: usize,
    pub(crate) group: usize,
    pub(crate) frame: Frame,
}

pub(crate) struct AliasPlan {
    pub(crate) heaps: Vec<Heap>,
    // Indexed by timeline.
    pub(crate) previous_use: Vec<PreviousUse>,
}

// One heap allocation: all lanes of one memory type.
pub(crate) struct Heap {
    pub(crate) members: Vec<HeapMember>,
}

pub(crate) enum HeapMember {
    Image {
        resource: u32,
        desc: ImageDescription,
        offset: u64,
        aliased: bool,
    },
    Buffer {
        resource: u32,
        desc: BufferDescription,
        offset: u64,
    },
}

struct Life {
    timeline: usize,
    resource: Resource,
    last_group: usize,
    start: usize,
    end: usize,
}

impl<'g, R, D> TaskGraph<'g, R, D> {
    pub(crate) fn build_alias_plan(&self, timelines: &[Timeline], schedule: &Schedule) -> AliasPlan {
        let mut previous_use = timelines
            .iter()
            .enumerate()
            .map(|(timeline, tl)| PreviousUse {
                timeline,
                group: tl.groups.len() - 1,
                frame: Frame::Previous,
            })
            .collect::<Vec<_>>();

        let heaps = self.build_heaps(timelines, schedule, &mut previous_use);
        AliasPlan {
            heaps,
            previous_use,
        }
    }

    fn build_heaps(&self, timelines: &[Timeline], schedule: &Schedule, previous_use: &mut [PreviousUse]) -> Vec<Heap> {
        let mut classes: BTreeMap<usize, Vec<Life>> = BTreeMap::new();
        for (timeline, tl) in timelines.iter().enumerate().filter(|(_, tl)| tl.transient) {
            assert!(is_write(&tl.groups[0].accesses), "transient resource {:?} must start with a write", tl.resource);

            let memory_type = self.transient_memory_type(tl.resource);
            classes.entry(memory_type as usize).or_default().push(Life {
                timeline,
                resource: tl.resource,
                last_group: tl.groups.len() - 1,
                start: schedule.span(timeline, 0).start(),
                end: schedule.span(timeline, tl.groups.len() - 1).end(),
            });
        }

        let mut heaps = Vec::with_capacity(classes.len());
        for (_, mut lives) in classes {
            lives.sort_by_key(|life| (life.start, life.timeline));

            let mut lanes: Vec<Vec<Life>> = Vec::new();
            for life in lives {
                match lanes.iter_mut().find(|lane| lane.last().expect("lane has owners").end < life.start) {
                    Some(lane) => lane.push(life),
                    None => lanes.push(vec![life]),
                }
            }

            let mut members = Vec::new();
            let mut watermark = 0u64;
            for lane in lanes {
                let image_count = lane.iter().filter(|life| matches!(life.resource, Resource::Image(_))).count();
                let aliased = image_count >= 2;
                let requirements: Vec<ResourceRequirements> = lane.iter().map(|life| self.requirements(life.resource, aliased)).collect();
                let size = requirements.iter().map(|requirements| requirements.size).max().expect("lane has owners");
                let alignment = requirements.iter().map(|requirements| requirements.alignment).max().expect("lane has owners");

                let offset = watermark.div_ceil(alignment) * alignment;
                watermark = offset + size;

                for (index, life) in lane.iter().enumerate() {
                    let previous = if index == 0 {
                        lane.last().expect("lane has owners")
                    } else {
                        &lane[index - 1]
                    };
                    previous_use[life.timeline] = PreviousUse {
                        timeline: previous.timeline,
                        group: previous.last_group,
                        frame: if index == 0 {
                            Frame::Previous
                        } else {
                            Frame::Current
                        },
                    };

                    match life.resource {
                        Resource::Image(resource) => {
                            let InnerImage::Transient {
                                image_desc, ..
                            } = &self.images[resource as usize]
                            else {
                                unreachable!("lane owner is transient")
                            };
                            members.push(HeapMember::Image {
                                resource,
                                desc: *image_desc,
                                offset,
                                aliased,
                            });
                        }
                        Resource::Buffer(resource) => {
                            let InnerBuffer::Transient {
                                buffer_desc,
                                ..
                            } = &self.buffers[resource as usize]
                            else {
                                unreachable!("lane owner is transient")
                            };
                            members.push(HeapMember::Buffer {
                                resource,
                                desc: *buffer_desc,
                                offset,
                            });
                        }
                    }
                }
            }

            heaps.push(Heap { members });
        }

        heaps
    }

    fn transient_memory_type(&self, resource: Resource) -> MemoryType {
        match resource {
            Resource::Image(resource) => {
                let InnerImage::Transient {
                    image_desc, ..
                } = &self.images[resource as usize]
                else {
                    unreachable!("timeline marked transient")
                };
                image_desc.memory_type
            }
            Resource::Buffer(resource) => {
                let InnerBuffer::Transient {
                    buffer_desc,
                    ..
                } = &self.buffers[resource as usize]
                else {
                    unreachable!("timeline marked transient")
                };
                buffer_desc.memory_type
            }
        }
    }

    fn requirements(&self, resource: Resource, aliased: bool) -> ResourceRequirements {
        match resource {
            Resource::Image(resource) => {
                let InnerImage::Transient {
                    image_desc, ..
                } = &self.images[resource as usize]
                else {
                    unreachable!("lane owner is transient")
                };
                sgpu::image_requirements(image_desc, aliased)
            }
            Resource::Buffer(resource) => {
                let InnerBuffer::Transient {
                    buffer_desc,
                    ..
                } = &self.buffers[resource as usize]
                else {
                    unreachable!("lane owner is transient")
                };
                sgpu::buffer_requirements(buffer_desc)
            }
        }
    }

    // Rebuild all transient instances from the alias plan.
    // The caller must have waited for the GPU before calling.
    pub(crate) fn materialize_transients(&mut self, plan: &AliasPlan) {
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

        for heap in &plan.heaps {
            let mut image_inputs: Vec<PooledImageDescription> = Vec::new();
            let mut image_racks: Vec<u32> = Vec::new();
            let mut buffer_inputs: Vec<PooledBufferDescription> = Vec::new();
            let mut buffer_racks: Vec<u32> = Vec::new();

            for member in &heap.members {
                match member {
                    HeapMember::Image {
                        resource,
                        desc,
                        offset,
                        aliased,
                    } => {
                        image_inputs.push(PooledImageDescription {
                            desc: *desc,
                            offset: *offset,
                            aliased: *aliased,
                        });
                        image_racks.push(*resource);
                    }
                    HeapMember::Buffer {
                        resource,
                        desc,
                        offset,
                    } => {
                        buffer_inputs.push(PooledBufferDescription {
                            desc: *desc,
                            offset: *offset,
                        });
                        buffer_racks.push(*resource);
                    }
                }
            }

            for slot in 0..slots {
                let (slot_images, slot_buffers) = unsafe {
                    sgpu::create_pooled_resources(&PooledResourceDescription {
                        images: &image_inputs,
                        buffers: &buffer_inputs,
                    })
                };

                for (member_index, rack) in image_racks.iter().enumerate() {
                    let InnerImage::Transient { images, .. } = &mut self.images[*rack as usize] else {
                        unreachable!("transient member above")
                    };
                    debug_assert_eq!(images.len(), slot, "rack {rack} instance sets drifted between slots");
                    images.push(slot_images[member_index]);
                }
                for (member_index, rack) in buffer_racks.iter().enumerate() {
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

            debug_assert!(image_racks.iter().all(|&rack| { matches!(&self.images[rack as usize], InnerImage::Transient { images, .. } if images.len() == slots) }));
            debug_assert!(buffer_racks.iter().all(|&rack| { matches!(&self.buffers[rack as usize], InnerBuffer::Transient { buffers, .. } if buffers.len() == slots) }));
        }
    }
}
