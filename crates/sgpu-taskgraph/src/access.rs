use sgpu::{AccessType, Read, Usage, Write};

pub trait Access {
    const ACCESS: AccessType;
    type U: Usage;
}

/// Marks an [`Access`] as valid for **images**: usable with
/// [`crate::NodeWriter::image`] / [`crate::ExecutionContext::image`]
/// (the only way to obtain a [`crate::PassImage`]).
pub trait ImageAccess: Access {}

/// Marks an [`Access`] as valid for **buffers**. See [`ImageAccess`].
pub trait BufferAccess: Access {}

/// Marker traits that gate which ID accessors a [`crate::ExecutionContext`] exposes
/// for a [`crate::PassImage`].
/// A color attachment cannot yield a storage ID; a sampled image cannot either.
pub trait StorageImageAccess: ImageAccess {}
pub trait SampledImageAccess: ImageAccess {}

// ---------------------------------------------------------------------------------------
// Shared (image + buffer) markers — one `Access` impl covers both kinds
// ---------------------------------------------------------------------------------------

/// Compute storage write (`ReadWriteBuffer` / writable storage image).
pub struct ComputeWrite;
impl Access for ComputeWrite {
    const ACCESS: AccessType = AccessType::ComputeShaderStorageWrite;
    type U = Write;
}
impl ImageAccess for ComputeWrite {}
impl BufferAccess for ComputeWrite {}
impl StorageImageAccess for ComputeWrite {}

/// Compute storage read.
pub struct ComputeRead;
impl Access for ComputeRead {
    const ACCESS: AccessType = AccessType::ComputeShaderStorageRead;
    type U = Read;
}
impl ImageAccess for ComputeRead {}
impl BufferAccess for ComputeRead {}
impl StorageImageAccess for ComputeRead {}

/// Generic any-shader storage write (stage-agnostic, conservative).
pub struct StorageWrite;
impl Access for StorageWrite {
    const ACCESS: AccessType = AccessType::AnyShaderStorageWrite;
    type U = Write;
}
impl ImageAccess for StorageWrite {}
impl BufferAccess for StorageWrite {}
impl StorageImageAccess for StorageWrite {}

/// Generic any-shader storage read.
pub struct StorageRead;
impl Access for StorageRead {
    const ACCESS: AccessType = AccessType::AnyShaderStorageRead;
    type U = Read;
}
impl ImageAccess for StorageRead {}
impl BufferAccess for StorageRead {}
impl StorageImageAccess for StorageRead {}

// ---------------------------------------------------------------------------------------
// Transfer markers (image + buffer)
// ---------------------------------------------------------------------------------------

/// Transfer write (copy / blit / fill destination).
pub struct TransferWrite;
impl Access for TransferWrite {
    const ACCESS: AccessType = AccessType::TransferWrite;
    type U = Write;
}
impl ImageAccess for TransferWrite {}
impl BufferAccess for TransferWrite {}

/// Transfer read (copy / blit source).
pub struct TransferRead;
impl Access for TransferRead {
    const ACCESS: AccessType = AccessType::TransferRead;
    type U = Read;
}
impl ImageAccess for TransferRead {}
impl BufferAccess for TransferRead {}

// ---------------------------------------------------------------------------------------
// Sampled image markers
// ---------------------------------------------------------------------------------------

/// Sampled read in a compute shader.
pub struct ComputeSampledRead;
impl Access for ComputeSampledRead {
    const ACCESS: AccessType = AccessType::ComputeShaderSampledRead;
    type U = Read;
}
impl ImageAccess for ComputeSampledRead {}
impl SampledImageAccess for ComputeSampledRead {}

/// Sampled read in a fragment shader.
pub struct FragmentSampledRead;
impl Access for FragmentSampledRead {
    const ACCESS: AccessType = AccessType::FragmentShaderSampledRead;
    type U = Read;
}
impl ImageAccess for FragmentSampledRead {}
impl SampledImageAccess for FragmentSampledRead {}

// ---------------------------------------------------------------------------------------
// Render-pass markers (images)
// ---------------------------------------------------------------------------------------

/// Storage write in a fragment shader.
pub struct FragmentWrite;
impl Access for FragmentWrite {
    const ACCESS: AccessType = AccessType::FragmentShaderStorageWrite;
    type U = Write;
}
impl ImageAccess for FragmentWrite {}
impl StorageImageAccess for FragmentWrite {}

/// Color attachment write (blend / raster output).
pub struct ColorWrite;
impl Access for ColorWrite {
    const ACCESS: AccessType = AccessType::ColorAttachmentWrite;
    type U = Write;
}
impl ImageAccess for ColorWrite {}

/// Depth (or depth-stencil) attachment write.
pub struct DepthWrite;
impl Access for DepthWrite {
    const ACCESS: AccessType = AccessType::DepthStencilAttachmentWrite;
    type U = Write;
}
impl ImageAccess for DepthWrite {}

/// Final transition to a presentable layout. Record nothing; the graph emits
/// the `-> Present` barrier. Present the image outside the graph using the
/// [`sgpu::Counter`] returned by [`crate::Graph::execute`].
pub struct Present;
impl Access for Present {
    const ACCESS: AccessType = AccessType::Present;
    type U = Read;
}
impl ImageAccess for Present {}

// ---------------------------------------------------------------------------------------
// Buffer-only markers
// ---------------------------------------------------------------------------------------

/// Uniform buffer read (any shader stage).
pub struct UniformRead;
impl Access for UniformRead {
    const ACCESS: AccessType = AccessType::AnyShaderUniformRead;
    type U = Read;
}
impl BufferAccess for UniformRead {}

/// Vertex buffer read (`vkCmdBindVertexBuffers` input).
pub struct VertexRead;
impl Access for VertexRead {
    const ACCESS: AccessType = AccessType::VertexBuffer;
    type U = Read;
}
impl BufferAccess for VertexRead {}

/// Index buffer read.
pub struct IndexRead;
impl Access for IndexRead {
    const ACCESS: AccessType = AccessType::IndexBuffer;
    type U = Read;
}
impl BufferAccess for IndexRead {}

/// Indirect-draw argument buffer read.
pub struct IndirectRead;
impl Access for IndirectRead {
    const ACCESS: AccessType = AccessType::IndirectBuffer;
    type U = Read;
}
impl BufferAccess for IndirectRead {}
