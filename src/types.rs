use ash::vk;
use gpu_allocator::MemoryLocation;
use slotmap::DefaultKey;

use crate::commands::QueueType;

/// Memory Types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryType {
    /// Resides on Gpu, Not accessible by the host
    DeviceLocal,
    /// Resides on Host, accessible by Gpu
    PreferHost,
    /// Resides on Gpu, accessible by Host
    HostVisible,
    /// Let driver choose
    Auto,
}

impl MemoryType {
    pub(crate) const fn to_vk_flag(&self) -> MemoryLocation {
        match self {
            Self::DeviceLocal => MemoryLocation::GpuOnly,
            Self::PreferHost => MemoryLocation::CpuToGpu,
            Self::HostVisible => MemoryLocation::GpuToCpu,
            Self::Auto => MemoryLocation::Unknown,
        }
    }
}

bitflags::bitflags! {
    /// A wrapper struct for Vulkan's buffer usage flags.
    /// Can be combined using Bitwise Or (|)
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferUsage: u32 {
        const STORAGE = vk::BufferUsageFlags::STORAGE_BUFFER.as_raw();
        const VERTEX = vk::BufferUsageFlags::VERTEX_BUFFER.as_raw();
        const INDEX = vk::BufferUsageFlags::INDEX_BUFFER.as_raw();
        const UNIFORM = vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw();
        const INDIRECT = vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw();
        const TRANSFER_SRC = vk::BufferUsageFlags::TRANSFER_SRC.as_raw();
        const TRANSFER_DST = vk::BufferUsageFlags::TRANSFER_DST.as_raw();
    }
}

impl BufferUsage {
    #[inline]
    pub const fn to_vk(self) -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::from_raw(self.bits())
    }
}

/// Buffer descriptions, create mapped works only for perfer host memory type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDescription {
    pub usage: BufferUsage,
    pub size: vk::DeviceSize,
    pub memory_type: MemoryType,
}

#[derive(Clone, Copy)]
pub struct Buffer {
    pub(crate) id: DefaultKey,
}

impl Buffer {
    pub fn as_slice<T>(&self) -> &[T] {
        let inner = crate::CONTEXT.get().unwrap().get_buffer_inner(self);

        return unsafe {
            std::slice::from_raw_parts(inner.mapped_ptr.expect("Buffer needs to be host visible for slice") as *const T, inner.mem_requirements.size as usize / std::mem::size_of::<T>())
        };
    }
}

/// Used for sync. Every submit returns a counter
/// Other command buffers can wait on a counter.
// u8 queue + u56 counter
#[derive(Clone, Copy)]
pub struct Counter {
    data: u64,
}

impl Counter {
    const VALUE_MASK: u64 = (1u64 << 56) - 1;
    const QUEUE_MASK: u64 = !Self::VALUE_MASK;

    pub(crate) fn encode(queue: QueueType, value: u64) -> Counter {
        let q = (queue as u64) << 56;
        return Counter {
            data: q | (value & Self::VALUE_MASK),
        };
    }

    pub(crate) fn decode(&self) -> (QueueType, u64) {
        let queue_raw = (self.data >> 56) as usize;
        let value = self.data & Self::VALUE_MASK;

        let queue = QueueType::QUEUE_TYPES[queue_raw];
        (queue, value)
    }
}
