use std::sync::atomic::{AtomicU32, Ordering};

use sgpu::*;
use smallvec::SmallVec;

use super::compile::alias::{Heap, HeapMember};
use crate::resource::{PassBufferId, PassImageId};

pub(crate) enum InnerImage {
    //External Image supplied by the user
    External(Image),
    // Graph created persistent image
    // 1 image per frame in flight
    Persistent {
        image_desc: ImageDescription,
        images: Vec<Image>,
        resize: bool,
    },
    // Graph created transient image
    // 1 image per frame in flight
    // The memory of the image might be shared with outher resources
    Transient {
        image_desc: ImageDescription,
        images: Vec<Image>,
        resize: bool,
    },
}

impl InnerImage {
    pub(crate) fn image_at(&self, frame_index: usize) -> Image {
        match self {
            Self::External(image) => *image,
            Self::Persistent { images, .. } | Self::Transient { images, .. } => *images.get(frame_index).unwrap_or_else(|| panic!("slotted rack has no instance for slot {frame_index}")),
        }
    }

    pub(crate) fn is_graph_owned(&self) -> bool { matches!(self, Self::Persistent { .. } | Self::Transient { .. }) }

    pub(crate) fn is_transient(&self) -> bool { matches!(self, Self::Transient { .. }) }
}

pub(crate) enum InnerBuffer {
    //External Buffer supplied by the user
    External(Buffer),
    // Graph created persistent buffer
    // 1 image per frame in flight
    Persistent {
        buffer_desc: BufferDescription,
        buffers: Vec<Buffer>,
    },
    // Graph created transient buffer
    // 1 image per frame in flight
    // The memory of the buffer might be shared with outher resources
    Transient {
        buffer_desc: BufferDescription,
        buffers: Vec<Buffer>,
    },
}

impl InnerBuffer {
    // Get underlying buffer at a given frame
    pub(crate) fn buffer_at(&self, frame_index: usize) -> Buffer {
        match self {
            Self::External(buffer) => *buffer,
            Self::Persistent {
                buffers, ..
            }
            | Self::Transient {
                buffers, ..
            } => *buffers.get(frame_index).unwrap_or_else(|| panic!("slotted buffer rack has no instance for slot {frame_index}")),
        }
    }

    pub(crate) fn is_transient(&self) -> bool { matches!(self, Self::Transient { .. }) }
}

const MAX_VIEWS_PER_SPEC: usize = 8;
const RETIRE_DRAIN_AT: usize = 16;

struct ViewEntry {
    image: Image,
    view: ImageView,
}

struct ImageViewSpecification {
    image_id: u32,
    image_view_desc: ImageViewDescription,
    entries: SmallVec<[ViewEntry; 4]>,
}

/// View objects, deduplicated per `(image, description)` spec.
#[derive(Default)]
struct ImageViewPool {
    specs: Vec<ImageViewSpecification>,
    retired: Vec<ImageView>,
}

impl ImageViewPool {
    fn register(&mut self, image_id: u32, desc: ImageViewDescription) -> u32 {
        if let Some(i) = self.specs.iter().position(|s| s.image_id == image_id && s.image_view_desc == desc) {
            return i as u32;
        }

        self.specs.push(ImageViewSpecification {
            image_id,
            image_view_desc: desc,
            entries: SmallVec::new(),
        });
        self.specs.len() as u32 - 1
    }

    // moves items into retired queue
    fn destroy_slot(&mut self, image_id: u32) {
        for spec in &mut self.specs {
            if spec.image_id == image_id {
                self.retired.extend(spec.entries.drain(..).map(|entry| entry.view));
            }
        }
    }

    fn refresh(&mut self, images: &[InnerImage]) {
        for spec in &mut self.specs {
            match &images[spec.image_id as usize] {
                InnerImage::External(image) => {
                    if *image == Image::default() {
                        continue;
                    }
                    if spec.entries.iter().any(|e| e.image == *image) {
                        continue;
                    }
                    let evicted = (spec.entries.len() == MAX_VIEWS_PER_SPEC).then(|| spec.entries.remove(0).view);
                    spec.entries.push(ViewEntry {
                        image: *image,
                        view: image.create_view(&spec.image_view_desc),
                    });
                    if let Some(view) = evicted {
                        self.retired.push(view);
                    }
                }
                InnerImage::Persistent { images, .. } | InnerImage::Transient { images, .. } => {
                    for image in images {
                        if *image == Image::default() {
                            continue;
                        }
                        if spec.entries.iter().any(|e| e.image == *image) {
                            continue;
                        }
                        let evicted = (spec.entries.len() == MAX_VIEWS_PER_SPEC).then(|| spec.entries.remove(0).view);
                        spec.entries.push(ViewEntry {
                            image: *image,
                            view: image.create_view(&spec.image_view_desc),
                        });
                        if let Some(view) = evicted {
                            self.retired.push(view);
                        }
                    }
                }
            }
        }
        if self.retired.len() >= RETIRE_DRAIN_AT {
            self.drain();
        }
    }

    // destroys all the views
    fn destroy_all(&mut self) {
        if self.specs.is_empty() && self.retired.is_empty() {
            return;
        }
        sgpu::wait_idle();

        for spec in &mut self.specs {
            for entry in spec.entries.drain(..) {
                destroy_image_view(entry.view);
            }
        }
        for view in self.retired.drain(..) {
            destroy_image_view(view);
        }
    }

    // clears the retired image views
    fn drain(&mut self) {
        if self.retired.is_empty() {
            return;
        }
        sgpu::wait_idle();
        for view in self.retired.drain(..) {
            destroy_image_view(view);
        }
    }
}

/// The graph's resource storage: every image, buffer and view it owns, plus
/// the logic that creates, resizes, materializes and destroys them.
pub(crate) struct ResourcePool {
    graph_id: u32,
    pub(crate) images: Vec<InnerImage>,
    pub(crate) buffers: Vec<InnerBuffer>,
    views: ImageViewPool,
}

impl ResourcePool {
    pub(crate) fn new() -> Self {
        ResourcePool {
            graph_id: next_graph_id(),
            images: Vec::new(),
            buffers: Vec::new(),
            views: ImageViewPool::default(),
        }
    }

    pub(crate) fn push_image(&mut self, image: InnerImage) -> PassImageId {
        let id = PassImageId::new(self.images.len() as u32, self.graph_id);
        self.images.push(image);
        id
    }

    pub(crate) fn push_buffer(&mut self, buffer: InnerBuffer) -> PassBufferId {
        let id = PassBufferId::new(self.buffers.len() as u32, self.graph_id);
        self.buffers.push(buffer);
        id
    }

    pub(crate) fn check_image_raw(&self, id: PassImageId) -> u32 { id.check_graph(self.graph_id) }

    pub(crate) fn check_buffer_raw(&self, id: PassBufferId) -> u32 { id.check_graph(self.graph_id) }

    pub(crate) fn get_image(&self, id: PassImageId) -> &InnerImage { &self.images[id.check_graph(self.graph_id) as usize] }

    pub(crate) fn get_buffer(&self, id: PassBufferId) -> &InnerBuffer { &self.buffers[id.check_graph(self.graph_id) as usize] }

    /// Rebind an external image
    pub(crate) fn set_image(&mut self, id: PassImageId, image: Image) {
        match self.images[id.0 as usize] {
            InnerImage::External(_) => self.images[id.0 as usize] = InnerImage::External(image),
            _ => panic!("image {id:?} is not a plain external image and cannot be rebound"),
        }
    }

    /// Rebind an external buffer
    pub(crate) fn set_buffer(&mut self, id: PassBufferId, buffer: Buffer) {
        match self.buffers[id.0 as usize] {
            InnerBuffer::External(_) => self.buffers[id.0 as usize] = InnerBuffer::External(buffer),
            _ => panic!("buffer {id:?} is not a plain external buffer and cannot be rebound"),
        }
    }

    pub(crate) fn register_view(&mut self, image_id: u32, desc: ImageViewDescription) -> u32 { self.views.register(image_id, desc) }

    /// The view of `image_id`'s instance in `slot`, creating it on first use.
    pub(crate) fn view(&self, image_id: u32, slot: usize) -> ImageView {
        let spec = &self.views.specs[image_id as usize];
        let image = self.images[spec.image_id as usize].image_at(slot);
        if image == Image::default() {
            panic!("image view requested for unbound swapchain slot {slot}");
        }
        spec.entries.iter().find(|e| e.image == image).expect("image view not created (image unbound?)").view
    }

    /// Creates the views that are missing for the current instances of every
    /// registered image, retiring views of evicted instances.
    pub(crate) fn refresh_views(&mut self) { self.views.refresh(&self.images); }

    /// The image behind `index` in frame slot `slot`
    pub(crate) fn image_at(&self, index: u32, slot: usize) -> Image { self.images[index as usize].image_at(slot) }

    /// The buffer behind `index` in frame slot `slot`
    pub(crate) fn buffer_at(&self, index: u32, slot: usize) -> Buffer { self.buffers[index as usize].buffer_at(slot) }

    /// Point every resizable image at a new extent and recreate the persistant images
    pub(crate) fn resize_images(&mut self, width: u32, height: u32, slots: usize) {
        for img in &mut self.images {
            match img {
                InnerImage::Persistent {
                    image_desc,
                    resize: true,
                    ..
                }
                | InnerImage::Transient {
                    image_desc,
                    resize: true,
                    ..
                } => {
                    image_desc.extent.width = width;
                    image_desc.extent.height = height;
                }
                _ => {}
            }
        }

        self.materialize_persistent_images(slots, true);
    }

    pub(crate) fn materialize_persistent_images(&mut self, slots: usize, recreate_resize: bool) {
        let mut old_images: Vec<(usize, Vec<Image>)> = Vec::new();
        for (index, rack) in self.images.iter_mut().enumerate() {
            if let InnerImage::Persistent {
                images,
                resize,
                ..
            } = rack
                && (images.len() != slots || (recreate_resize && *resize))
            {
                old_images.push((index, std::mem::take(images)));
            }
        }

        for (index, images) in old_images {
            if images.is_empty() {
                continue;
            }
            self.views.destroy_slot(index as u32);
            for image in images {
                destroy_image(image);
            }
        }

        for rack in &mut self.images {
            if let InnerImage::Persistent {
                image_desc,
                images,
                ..
            } = rack
                && images.is_empty()
            {
                *images = (0..slots).map(|_| create_image(image_desc)).collect();
            }
        }
    }

    pub(crate) fn materialize_persistent_buffers(&mut self, slots: usize) {
        let mut old_buffers: Vec<Vec<Buffer>> = Vec::new();
        for buffer in &mut self.buffers {
            if let InnerBuffer::Persistent {
                buffers, ..
            } = buffer
                && buffers.len() != slots
            {
                old_buffers.push(std::mem::take(buffers));
            }
        }

        for buffers in old_buffers {
            for buffer in buffers {
                destroy_buffer(buffer);
            }
        }

        for rack in &mut self.buffers {
            if let InnerBuffer::Persistent {
                buffer_desc,
                buffers,
            } = rack
                && buffers.is_empty()
            {
                *buffers = (0..slots).map(|_| create_buffer(buffer_desc)).collect();
            }
        }
    }

    pub(crate) fn materialize_transients(&mut self, slots: usize, heaps: &[Heap]) {
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

        for heap in heaps {
            let mut image_inputs: Vec<PooledImageDescription> = Vec::new();
            let mut image_racks: Vec<PassImageId> = Vec::new();
            let mut buffer_inputs: Vec<PooledBufferDescription> = Vec::new();
            let mut buffer_racks: Vec<PassBufferId> = Vec::new();
            let mut watermark = 0u64;

            for lane in &heap.alias_grps {
                let requirements: Vec<ResourceRequirements> = lane.members.iter().map(|member| member_requirements(self, member, lane.aliased)).collect();
                let size = requirements.iter().map(|requirements| requirements.size).max().expect("lane has members");
                let alignment = requirements.iter().map(|requirements| requirements.alignment).max().expect("lane has members");

                let offset = watermark.div_ceil(alignment) * alignment;
                watermark = offset + size;

                for member in &lane.members {
                    match member {
                        HeapMember::Image(id) => {
                            let InnerImage::Transient {
                                image_desc, ..
                            } = self.get_image(*id)
                            else {
                                unreachable!("transient member above")
                            };
                            image_inputs.push(PooledImageDescription {
                                desc: *image_desc,
                                offset,
                                aliased: lane.aliased,
                            });
                            image_racks.push(*id);
                        }
                        HeapMember::Buffer(id) => {
                            let InnerBuffer::Transient {
                                buffer_desc, ..
                            } = self.get_buffer(*id)
                            else {
                                unreachable!("transient member above")
                            };
                            buffer_inputs.push(PooledBufferDescription {
                                desc: *buffer_desc,
                                offset,
                            });
                            buffer_racks.push(*id);
                        }
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
                    let InnerImage::Transient { images, .. } = &mut self.images[rack.0 as usize] else {
                        unreachable!("transient member above")
                    };
                    debug_assert_eq!(images.len(), slot, "rack {rack:?} instance sets drifted between slots");
                    images.push(slot_images[member_index]);
                }
                for (member_index, rack) in buffer_racks.iter().enumerate() {
                    let InnerBuffer::Transient {
                        buffers, ..
                    } = &mut self.buffers[rack.0 as usize]
                    else {
                        unreachable!("transient member above")
                    };
                    debug_assert_eq!(buffers.len(), slot, "rack {rack:?} instance sets drifted between slots");
                    buffers.push(slot_buffers[member_index]);
                }
            }
        }
    }
}

impl Drop for ResourcePool {
    fn drop(&mut self) {
        sgpu::wait_idle();
        self.views.destroy_all();
        for rack in &self.images {
            match rack {
                InnerImage::Persistent { images, .. } | InnerImage::Transient { images, .. } => {
                    images.iter().for_each(|img| destroy_image(*img));
                }
                _ => {}
            }
        }
        for rack in self.buffers.iter() {
            match rack {
                InnerBuffer::Persistent {
                    buffers, ..
                }
                | InnerBuffer::Transient {
                    buffers, ..
                } => {
                    buffers.iter().for_each(|buffer| destroy_buffer(*buffer));
                }
                _ => {}
            }
        }
    }
}

fn member_requirements(pool: &ResourcePool, member: &HeapMember, aliased: bool) -> ResourceRequirements {
    match member {
        HeapMember::Image(id) => {
            let InnerImage::Transient {
                image_desc, ..
            } = pool.get_image(*id)
            else {
                unreachable!("heap members are transients")
            };
            sgpu::image_requirements(image_desc, aliased)
        }
        HeapMember::Buffer(id) => {
            let InnerBuffer::Transient {
                buffer_desc, ..
            } = pool.get_buffer(*id)
            else {
                unreachable!("heap members are transients")
            };
            sgpu::buffer_requirements(buffer_desc)
        }
    }
}

static NEXT_GRAPH_ID: AtomicU32 = AtomicU32::new(1);

pub(crate) fn next_graph_id() -> u32 { NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed) }
