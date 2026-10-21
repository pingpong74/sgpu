use crate::{
    Buffer, BufferDescription, BufferUsage, ComputePipeline, Counter, Image, ImageDescription, ImageUsage, ImageView, ImageViewDescription, MemoryType, PooledResourceDescription, ResourceRequirements, Sampler, SamplerDescription,
    api::SgpuInititizationInfo,
    backend::{
        commands::*,
        descriptors::BindlessDescriptorSet,
        device::Device,
        resources::{InnerBuffer, InnerImage, InnerImageView},
    },
    commands::{CommandBuffer, QueueType},
    pipeline::{RasterizationPipeline, RasterizationPipelineDescription, ShaderModule},
    query::QueryPool,
};

use super::instance::Instance;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use slotmap::{DefaultKey, Key, SlotMap};

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    thread::ThreadId,
};

pub(crate) struct Context {
    pub(crate) instance: Instance,
    pub(crate) device: Device,
    pub(crate) bindless_descriptor_set: BindlessDescriptorSet,

    pub(crate) queues: Queues,

    pub(crate) buffers: RwLock<SlotMap<DefaultKey, InnerBuffer>>,
    pub(crate) images: RwLock<SlotMap<DefaultKey, InnerImage>>,
    pub(crate) image_views: RwLock<SlotMap<DefaultKey, InnerImageView>>,
    pub(crate) samplers: RwLock<SlotMap<DefaultKey, vk::Sampler>>,

    pub(crate) thread_cmd_pool_pool: RwLock<HashMap<ThreadId, ThreadCommandPools>>,
}

impl Context {
    pub(crate) fn new(init_info: &SgpuInititizationInfo) -> Context {
        let instance = Instance::new(init_info);
        let device = Device::new(init_info, &instance);
        let queues = device.get_queues();
        let bindless_set = BindlessDescriptorSet::new(&device);

        Context {
            instance,
            device,
            queues,
            bindless_descriptor_set: bindless_set,

            buffers: RwLock::new(Default::default()),
            images: RwLock::new(Default::default()),
            image_views: RwLock::new(Default::default()),
            samplers: RwLock::new(Default::default()),

            thread_cmd_pool_pool: RwLock::new(HashMap::new()),
        }
    }

    pub(crate) fn create_buffer(&self, desc: &BufferDescription) -> Buffer {
        let inner_buffer = self.device.create_buffer(desc);
        let buffer = inner_buffer.buffer;
        let id = self.buffers.write().unwrap().insert(inner_buffer);
        // Janky stuff
        if desc.usage.contains(BufferUsage::STORAGE) {
            self.bindless_descriptor_set.write_buffer(&self.device, buffer, id.data().as_ffi() as u32);
        }

        Buffer {
            raw: buffer,
            id,
        }
    }

    pub fn as_slice<T>(&self, buffer: &Buffer) -> &[T] {
        assert!(std::mem::size_of::<T>() > 0, "size must be > 0");

        let guard = self.buffers.read().unwrap();
        let inner = guard.get(buffer.id).expect("Invalid Buffer");
        let mapped_ptr = inner.mapped_ptr.expect("Buffer needs to be host visible for slice");
        let len = inner.size as usize / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(mapped_ptr as *const T, len) }
    }

    // SAFETY: returns a slice over persistently mapped buffer memory.
    // The caller must not destroy the buffer while the slice is alive.
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_slice<T>(&self, buffer: &Buffer) -> &mut [T] {
        assert!(std::mem::size_of::<T>() > 0, "size must be > 0");

        let mut guard = self.buffers.write().unwrap();
        let inner = guard.get_mut(buffer.id).expect("Invalid Buffer");
        let mapped_ptr = inner.mapped_ptr.expect("Buffer needs to be host visible for slice");
        let len = inner.size as usize / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(mapped_ptr as *mut T, len) }
    }

    pub(crate) fn destroy_buffer(&self, buffer: Buffer) {
        let inner = self.buffers.write().unwrap().remove(buffer.id).expect("Tried destroy buffer twice");

        self.device.destroy_buffer(inner);
    }

    pub(crate) fn create_image(&self, desc: &ImageDescription) -> Image {
        let inner_image = self.device.create_image(desc);
        let raw_image = inner_image.image;
        let id = self.images.write().unwrap().insert(inner_image);

        let mut image = Image {
            raw: raw_image,
            default_view: ImageView::default(),
            id,
        };
        image.default_view = self.create_image_view(&desc.default_view, image);
        image
    }

    pub(crate) fn image_requirements(&self, desc: &ImageDescription) -> ResourceRequirements {
        let requirements = self.device.image_requirements(desc);
        ResourceRequirements {
            size: requirements.size,
            alignment: requirements.alignment,
            memory_type_bits: requirements.memory_type_bits,
        }
    }

    pub(crate) fn buffer_requirements(&self, desc: &BufferDescription) -> ResourceRequirements {
        let requirements = self.device.buffer_requirements(desc);
        ResourceRequirements {
            size: requirements.size,
            alignment: requirements.alignment,
            memory_type_bits: requirements.memory_type_bits,
        }
    }

    /// Creates one transient heap and binds a pooled instance per member at
    /// its scheduled offset. The heap's location, size, alignment and
    /// memory-type bits are derived from the members themselves — the
    /// object-less probes are description-deterministic, so the
    /// re-derivation reproduces the sweep's aggregates exactly. One heap +
    /// one instance set per call; per-slot heaps come from calling it once
    /// per slot.
    pub(crate) fn create_pooled_resources(&self, plan: &PooledResourceDescription<'_>) -> (Vec<Image>, Vec<Buffer>) {
        let mut memory_type: Option<MemoryType> = None;
        let mut alignment = 1u64;
        let mut memory_type_bits = u32::MAX;
        let mut end = 0u64;

        let members = plan.images.iter().map(|member| (member.desc.memory_type, self.device.image_requirements(&member.desc), member.offset)).chain(plan.buffers.iter().map(|member| (member.desc.memory_type, self.device.buffer_requirements(&member.desc), member.offset)));
        for (member_type, requirements, offset) in members {
            end = (offset + requirements.size).max(end);
            alignment = alignment.max(requirements.alignment);
            memory_type_bits &= requirements.memory_type_bits;
            match memory_type {
                Some(existing) => assert_eq!(existing, member_type, "pooled plan mixes memory types ({existing:?} vs {member_type:?})"),
                None => memory_type = Some(member_type),
            }
        }

        let Some(memory_type) = memory_type else {
            return (Vec::new(), Vec::new());
        };

        // The sweep's final claim end, rounded up to the aggregate alignment.
        let heap_size = end.div_ceil(alignment).max(1) * alignment;

        let linear = plan.images.is_empty();

        let allocation: Arc<Allocation> = Arc::new(
            self.device
                .allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: &format!("taskgraph-heap-{:?}", memory_type),
                    requirements: vk::MemoryRequirements {
                        size: heap_size,
                        alignment,
                        memory_type_bits,
                    },
                    location: memory_type.to_vk(),
                    linear,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .expect("Failed to allocate transient heap"),
        );

        let pooled_images: Vec<Image> = plan
            .images
            .iter()
            .map(|member| {
                let inner = self.device.create_pooled_image(&member.desc, allocation.clone(), member.offset, member.aliased);
                let raw = inner.image;
                let id = self.images.write().unwrap().insert(inner);

                let mut image = Image {
                    raw,
                    default_view: ImageView::default(),
                    id,
                };
                image.default_view = self.create_image_view(&member.desc.default_view, image);
                image
            })
            .collect();

        let pooled_buffers: Vec<Buffer> = plan
            .buffers
            .iter()
            .map(|member| {
                let inner = self.device.create_pooled_buffer(&member.desc, allocation.clone(), member.offset);
                let buffer = inner.buffer;
                let id = self.buffers.write().unwrap().insert(inner);

                if member.desc.usage.contains(BufferUsage::STORAGE) {
                    self.bindless_descriptor_set.write_buffer(&self.device, buffer, id.data().as_ffi() as u32);
                }

                Buffer {
                    raw: buffer,
                    id,
                }
            })
            .collect();

        (pooled_images, pooled_buffers)
    }

    pub(crate) fn create_image_view(&self, desc: &ImageViewDescription, image: Image) -> ImageView {
        let (inner_view, usage) = {
            let images = self.images.read().unwrap();
            let inner_image = images.get(image.id).expect("Invalid image Id");
            (self.device.create_image_view(inner_image, desc), inner_image.usage)
        };

        let raw = inner_view.view;

        let id = self.image_views.write().unwrap().insert(inner_view);

        if usage.contains(ImageUsage::SAMPLED) {
            self.bindless_descriptor_set.write_sampled_image(&self.device, raw, id.data().as_ffi() as u32);
        }

        if usage.contains(ImageUsage::STORAGE) {
            self.bindless_descriptor_set.write_storage_image(&self.device, raw, id.data().as_ffi() as u32);
        }

        return ImageView { raw, id };
    }

    pub(crate) fn create_sampler(&self, desc: &SamplerDescription) -> Sampler {
        let sampler = self.device.create_sampler(desc);
        let id = self.samplers.write().unwrap().insert(sampler);
        self.bindless_descriptor_set.write_sampler(&self.device, sampler, id.data().as_ffi() as u32);

        Sampler {
            raw: sampler,
            id,
        }
    }

    pub(crate) fn destroy_sampler(&self, sampler: Sampler) {
        self.samplers.write().unwrap().remove(sampler.id).expect("Tried to destroy Sampler twice");
        self.device.destroy_sampler(sampler.raw);
    }

    pub(crate) fn destroy_pipeline(&self, pipeline: ash::vk::Pipeline) { unsafe { self.device.handle.destroy_pipeline(pipeline, None) }; }

    pub(crate) fn destroy_image(&self, image: Image) {
        let inner = self.images.write().unwrap().remove(image.id).expect("Tried to destroy Image twice");

        if let Some(view) = self.image_views.write().unwrap().remove(image.default_view.id) {
            self.device.destroy_image_view(view);
        }

        self.device.destroy_image(inner);
    }

    pub(crate) fn destroy_image_view(&self, image_view: ImageView) {
        if let Some(inner) = self.image_views.write().unwrap().remove(image_view.id) {
            self.device.destroy_image_view(inner);
        }
    }

    pub(crate) fn create_raster_pipeline(&self, dec: &RasterizationPipelineDescription) -> RasterizationPipeline { self.device.create_raster_pipeline(&self.bindless_descriptor_set, dec) }

    pub(crate) fn create_compute_pipeline(&self, module: &ShaderModule) -> ComputePipeline { self.device.create_compute_pipeline(&self.bindless_descriptor_set, module) }

    pub(crate) fn create_query_pool(&self, query_count: u32) -> QueryPool {
        QueryPool {
            raw: self.device.create_query_pool(query_count),
            query_count,
        }
    }

    pub(crate) fn poll(&self, counter: Counter) -> bool {
        let (queue_type, value) = counter.decode();

        self.queues.get_queue(queue_type).poll(&self.device, value)
    }

    pub(crate) fn wait_idle(&self) { self.device.wait_idle(); }

    pub(crate) fn wait(&self, counter: Counter) {
        let (queue_type, value) = counter.decode();
        self.device.wait_queue(self.queues.get_queue(queue_type), value);
    }

    pub(crate) fn record(&'static self, queue_type: QueueType) -> CommandBuffer {
        let thread_id = std::thread::current().id();
        return self.thread_cmd_pool_pool.write().unwrap().entry(thread_id).or_insert(self.device.create_thread_command_pools(self)).record(queue_type, self);
    }

    pub(crate) fn submit(&self, command_buffers: &[CommandBuffer]) -> Counter {
        let queue_type = command_buffers[0].queue;

        debug_assert!(command_buffers.iter().all(|c| c.queue == queue_type), "All command buffers must be from the same queue");

        let counter = self.queues.get_queue(queue_type).submit(&self.device, command_buffers);

        // Mark all the pools to be free
        let thread_id = std::thread::current().id();
        let mut pools = self.thread_cmd_pool_pool.write().unwrap();
        if let Some(thread_pools) = pools.get_mut(&thread_id) {
            for cmd in command_buffers {
                thread_pools.mark_submitted(queue_type, cmd.pool_idx, counter);
            }
        }

        counter
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.image_views.write().unwrap().drain().for_each(|(_, view)| self.device.destroy_image_view(view));
        self.buffers.write().unwrap().drain().for_each(|(_, buffer)| self.device.destroy_buffer(buffer));
        self.images.write().unwrap().drain().for_each(|(_, img)| self.device.destroy_image(img));
        self.samplers.write().unwrap().drain().for_each(|(_, sampler)| self.device.destroy_sampler(sampler));

        self.bindless_descriptor_set.cleanup(&self.device);
        self.device.cleanup();
        self.instance.cleanup();
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}
