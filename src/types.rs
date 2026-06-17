use ash::vk;
use gpu_allocator::MemoryLocation;
use slotmap::{DefaultKey, Key};

use crate::commands::QueueType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Extent3D {
    pub(crate) fn to_vk(&self) -> ash::vk::Extent3D {
        return ash::vk::Extent3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
        };
    }
}

/// Wrapper for vk::Extent2D
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

impl Extent2D {
    pub(crate) fn to_vk(&self) -> ash::vk::Extent2D {
        return ash::vk::Extent2D {
            width: self.width,
            height: self.height,
        };
    }
}

/// Wrapper for vk::Offset3D
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Offset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Offset3D {
    pub(crate) fn to_vk(&self) -> ash::vk::Offset3D {
        return ash::vk::Offset3D {
            x: self.x,
            y: self.y,
            z: self.z,
        };
    }
}

/// Wrapper for vk::Offset2D
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

impl Offset2D {
    pub(crate) fn to_vk(&self) -> ash::vk::Offset2D {
        return ash::vk::Offset2D {
            x: self.x,
            y: self.y,
        };
    }
}

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
    pub(crate) const fn to_vk(&self) -> MemoryLocation {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageType {
    Type1D,
    Type2D,
    Type3D,
}

impl ImageType {
    pub(crate) fn to_vk(&self) -> vk::ImageType {
        match self {
            Self::Type1D => vk::ImageType::TYPE_1D,
            Self::Type2D => vk::ImageType::TYPE_2D,
            Self::Type3D => vk::ImageType::TYPE_3D,
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ImageUsage: u32 {
        const TRANSFER_SRC = vk::ImageUsageFlags::TRANSFER_SRC.as_raw();
        const TRANSFER_DST = vk::ImageUsageFlags::TRANSFER_DST.as_raw();
        const SAMPLED = vk::ImageUsageFlags::SAMPLED.as_raw();
        const STORAGE = vk::ImageUsageFlags::STORAGE.as_raw();
        const COLOR_ATTACHMENT = vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw();
        const DEPTH_STENCIL_ATTACHMENT = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT.as_raw();
    }
}

impl ImageUsage {
    #[inline]
    pub const fn to_vk(self) -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::from_raw(self.bits())
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ImageCreateFlags: u32 {
        const NONE = 0;
        const CUBE_COMPATIBLE = vk::ImageCreateFlags::CUBE_COMPATIBLE.as_raw();
        const ARRAY_2D_COMPATIBLE = vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE.as_raw();
        const MUTABLE_FORMAT = vk::ImageCreateFlags::MUTABLE_FORMAT.as_raw();
        const ALIAS = vk::ImageCreateFlags::ALIAS.as_raw();
        const SPARSE_BINDING = vk::ImageCreateFlags::SPARSE_BINDING.as_raw();
        const SPARSE_RESIDENCY = vk::ImageCreateFlags::SPARSE_RESIDENCY.as_raw();
        const SPARSE_ALIASED = vk::ImageCreateFlags::SPARSE_ALIASED.as_raw();
        const BLOCK_TEXEL_VIEW_COMPATIBLE = vk::ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE.as_raw();
        const EXTENDED_USAGE = vk::ImageCreateFlags::EXTENDED_USAGE.as_raw();
        const PROTECTED = vk::ImageCreateFlags::PROTECTED.as_raw();
        const DISJOINT = vk::ImageCreateFlags::DISJOINT.as_raw();
        const CORNER_SAMPLED_NV = vk::ImageCreateFlags::CORNER_SAMPLED_NV.as_raw();
    }
}

impl ImageCreateFlags {
    #[inline]
    pub const fn to_vk(self) -> vk::ImageCreateFlags {
        vk::ImageCreateFlags::from_raw(self.bits())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Rgba8Unorm,
    Bgra8Unorm,
    Rgb565Unorm,
    R32Uint,
    Rgba8Uint,
    Rgba32Sint,
    Rgba16Float,
    Rg32Float,
    Rgb32Float,
    Rgba32Float,
    R32Float,
    D32Float,
    D24UnormS8Uint,
    D16Unorm,
    BC1RgbaUnorm,
    BC7Unorm,
}

impl Format {
    pub(crate) const fn to_vk(&self) -> vk::Format {
        return match self {
            Self::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            Self::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            Self::Rgb565Unorm => vk::Format::R5G6B5_UNORM_PACK16,
            Self::R32Uint => vk::Format::R32_UINT,
            Self::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
            Self::Rgba32Sint => vk::Format::R32G32B32A32_SINT,
            Self::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            Self::Rg32Float => vk::Format::R32G32_SFLOAT,
            Self::Rgb32Float => vk::Format::R32G32B32_SFLOAT,
            Self::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            Self::R32Float => vk::Format::R32_SFLOAT,
            Self::D32Float => vk::Format::D32_SFLOAT,
            Self::D24UnormS8Uint => vk::Format::D24_UNORM_S8_UINT,
            Self::D16Unorm => vk::Format::D16_UNORM,
            Self::BC1RgbaUnorm => vk::Format::BC1_RGBA_UNORM_BLOCK,
            Self::BC7Unorm => vk::Format::BC7_UNORM_BLOCK,
        };
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleCount {
    Type1,
    Type2,
    Type4,
    Type8,
    Type16,
    Type32,
    Type64,
}

impl SampleCount {
    pub(crate) const fn to_vk(&self) -> vk::SampleCountFlags {
        return match self {
            Self::Type1 => vk::SampleCountFlags::TYPE_1,
            Self::Type2 => vk::SampleCountFlags::TYPE_2,
            Self::Type4 => vk::SampleCountFlags::TYPE_4,
            Self::Type8 => vk::SampleCountFlags::TYPE_8,
            Self::Type16 => vk::SampleCountFlags::TYPE_16,
            Self::Type32 => vk::SampleCountFlags::TYPE_32,
            Self::Type64 => vk::SampleCountFlags::TYPE_64,
        };
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct ImageDescription {
    pub create_flags: ImageCreateFlags,
    pub usage: ImageUsage,
    pub format: Format,
    pub image_type: ImageType,
    pub extent: Extent3D,
    pub memory_type: MemoryType,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: SampleCount,
    pub default_view: ImageViewDescription,
}

impl Default for ImageDescription {
    fn default() -> Self {
        return Self {
            create_flags: ImageCreateFlags::NONE,
            usage: ImageUsage::SAMPLED,
            format: Format::Rgba16Float,
            image_type: ImageType::Type2D,
            extent: Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            memory_type: MemoryType::Auto,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Type1,
            default_view: ImageViewDescription::default(),
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageSubresources {
    pub aspect: ImageAspect,
    pub mip_level: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,

    // this only matters for Subresource range
    pub level_count: u32,
}

impl ImageSubresources {
    pub(crate) const fn to_vk_subresource_layers(&self) -> vk::ImageSubresourceLayers {
        return vk::ImageSubresourceLayers {
            aspect_mask: self.aspect.to_vk(),
            mip_level: self.mip_level,
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count,
        };
    }

    pub(crate) const fn to_vk_subresource_range(&self) -> vk::ImageSubresourceRange {
        return vk::ImageSubresourceRange {
            aspect_mask: self.aspect.to_vk(),
            base_mip_level: self.mip_level,
            level_count: self.level_count,
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count,
        };
    }
}

impl Default for ImageSubresources {
    fn default() -> Self {
        return ImageSubresources {
            aspect: ImageAspect::COLOR,
            mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
    }
}

// Image View Description
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageViewType {
    Type1D,
    Type2D,
    Type3D,
    Cube,
    Type1DArray,
    Type2DArray,
    CubeArray,
}

impl ImageViewType {
    pub(crate) const fn to_vk(&self) -> vk::ImageViewType {
        match self {
            Self::Type1D => vk::ImageViewType::TYPE_1D,
            Self::Type2D => vk::ImageViewType::TYPE_2D,
            Self::Type3D => vk::ImageViewType::TYPE_3D,
            Self::Cube => vk::ImageViewType::CUBE,
            Self::Type1DArray => vk::ImageViewType::TYPE_1D_ARRAY,
            Self::Type2DArray => vk::ImageViewType::TYPE_2D_ARRAY,
            Self::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ImageAspect: u32 {
        const COLOR = vk::ImageAspectFlags::COLOR.as_raw();
        const DEPTH = vk::ImageAspectFlags::DEPTH.as_raw();
        const STENCIL = vk::ImageAspectFlags::STENCIL.as_raw();
    }
}

impl ImageAspect {
    #[inline]
    pub const fn to_vk(self) -> vk::ImageAspectFlags {
        vk::ImageAspectFlags::from_raw(self.bits())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageViewDescription {
    pub view_type: ImageViewType,
    pub subresources: ImageSubresources,
}

impl Default for ImageViewDescription {
    fn default() -> Self {
        return Self {
            view_type: ImageViewType::Type2D,
            subresources: ImageSubresources::default(),
        };
    }
}

/// A vulkan Buffer
#[derive(Clone, Copy, Default)]
pub struct Buffer {
    pub(crate) raw: vk::Buffer,
    pub(crate) id: DefaultKey,
}

impl Buffer {
    pub fn descriptor_index(&self) -> u32 {
        return self.id.data().as_ffi() as u32;
    }

    pub fn as_slice<T>(&self) -> &[T] {
        let inner = unsafe { crate::CONTEXT.get().unwrap().get_buffer_inner(self) };

        return unsafe { std::slice::from_raw_parts(inner.mapped_ptr.expect("Buffer needs to be host visible for slice") as *const T, inner.size as usize / std::mem::size_of::<T>()) };
    }

    pub fn as_mut_slice<T>(&self) -> &mut [T] {
        let inner = unsafe { crate::CONTEXT.get().unwrap().get_buffer_inner(self) };

        return unsafe { std::slice::from_raw_parts_mut(inner.mapped_ptr.expect("Buffer needs to be host visible for slice") as *mut T, inner.size as usize / std::mem::size_of::<T>()) };
    }
}

/// A Vulkan image, used to create and use image views
#[derive(Clone, Copy, Default)]
pub struct Image {
    pub(crate) raw: vk::Image,
    pub(crate) default_view: ImageView,
    pub(crate) id: DefaultKey,
}

impl Image {
    pub fn create_view(&self, image_view_desc: &ImageViewDescription) -> ImageView {
        let ctx = crate::CONTEXT.get().unwrap();
        return ctx.create_image_view(image_view_desc, *self);
    }

    pub fn default_view(&self) -> ImageView {
        return self.default_view;
    }
}

#[derive(Clone, Copy, PartialEq, Default)]
pub struct ImageView {
    pub(crate) raw: vk::ImageView,
    pub(crate) id: DefaultKey,
}

/// A region for buffer-to-image or image-to-buffer copy operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BufferImageCopyRegion {
    pub buffer_offset: u64,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_subresource: ImageSubresources,
    pub image_offset: Offset3D,
    pub image_extent: Extent3D,
}

impl BufferImageCopyRegion {
    pub(crate) fn to_vk(&self) -> vk::BufferImageCopy {
        vk::BufferImageCopy {
            buffer_offset: self.buffer_offset,
            buffer_row_length: self.buffer_row_length,
            buffer_image_height: self.buffer_image_height,
            image_subresource: self.image_subresource.to_vk_subresource_layers(),
            image_offset: self.image_offset.to_vk(),
            image_extent: self.image_extent.to_vk(),
            ..Default::default()
        }
    }
}

/// A region for image-to-image copy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageCopyRegion {
    pub src_subresource: ImageSubresources,
    pub src_offset: Offset3D,
    pub dst_subresource: ImageSubresources,
    pub dst_offset: Offset3D,
    pub extent: Extent3D,
}

impl ImageCopyRegion {
    pub(crate) fn to_vk(&self) -> vk::ImageCopy {
        vk::ImageCopy {
            src_subresource: self.src_subresource.to_vk_subresource_layers(),
            src_offset: self.src_offset.to_vk(),
            dst_subresource: self.dst_subresource.to_vk_subresource_layers(),
            dst_offset: self.dst_offset.to_vk(),
            extent: self.extent.to_vk(),
        }
    }
}

/// A region for blit operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlitRegion {
    pub src_subresource: ImageSubresources,
    pub src_offsets: [Offset3D; 2],
    pub dst_subresource: ImageSubresources,
    pub dst_offsets: [Offset3D; 2],
}

impl BlitRegion {
    pub(crate) fn to_vk(&self) -> vk::ImageBlit {
        vk::ImageBlit {
            src_subresource: self.src_subresource.to_vk_subresource_layers(),
            src_offsets: [self.src_offsets[0].to_vk(), self.src_offsets[1].to_vk()],
            dst_subresource: self.dst_subresource.to_vk_subresource_layers(),
            dst_offsets: [self.dst_offsets[0].to_vk(), self.dst_offsets[1].to_vk()],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Linear,
}

impl Filter {
    pub(crate) fn to_vk(&self) -> vk::Filter {
        match self {
            Self::Nearest => vk::Filter::NEAREST,
            Self::Linear => vk::Filter::LINEAR,
        }
    }
}

/// Used for sync. Every submit returns a counter
/// Other command buffers can wait on a counter.
/// u8 queue + u56 counter
#[derive(Clone, Copy)]
pub struct Counter {
    data: u64,
}

impl Counter {
    const VALUE_MASK: u64 = (1u64 << 56) - 1;

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
