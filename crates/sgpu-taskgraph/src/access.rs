use sgpu::{AccessType, Read, Usage, Write};

pub trait Access {
    const ACCESS: AccessType;
    type U: Usage;
}

pub trait ImageAccess: Access {}
pub trait BufferAccess: Access {}
pub trait StorageImageAccess: ImageAccess {}
pub trait SampledImageAccess: ImageAccess {}

///   declared as: type name = AccessType, Read/Write
///   storage =  shader storage on image, buffer and storage image (read/write)
///   storage_image = shader storage on image and storage image (read/write)
///   transfer = copy/blit/fill on image and buffer (read/write)
///   sampled = sampled read on an image
///   attachment = write-only image attachment
///   buffer = read-only buffer input (uniform/vertex/index/indirect)
macro_rules! marker {
    ($(#[$meta:meta])* storage $name:ident = $access:ident, $rw:ty) => {
        marker! { $(#[$meta])* $name: $access, $rw, ImageAccess, BufferAccess, StorageImageAccess }
    };
    ($(#[$meta:meta])* storage_image $name:ident = $access:ident, $rw:ty) => {
        marker! { $(#[$meta])* $name: $access, $rw, ImageAccess, StorageImageAccess }
    };
    ($(#[$meta:meta])* transfer $name:ident = $access:ident, $rw:ty) => {
        marker! { $(#[$meta])* $name: $access, $rw, ImageAccess, BufferAccess }
    };
    ($(#[$meta:meta])* sampled $name:ident = $access:ident) => {
        marker! { $(#[$meta])* $name: $access, Read, ImageAccess, SampledImageAccess }
    };
    ($(#[$meta:meta])* attachment $name:ident = $access:ident) => {
        marker! { $(#[$meta])* $name: $access, Write, ImageAccess }
    };
    ($(#[$meta:meta])* buffer $name:ident = $access:ident) => {
        marker! { $(#[$meta])* $name: $access, Read, BufferAccess }
    };
    ($(#[$meta:meta])* $name:ident: $access:ident, $rw:ty, $($impls:path),+ $(,)?) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name;
        impl Access for $name {
            const ACCESS: AccessType = AccessType::$access;
            type U = $rw;
        }
        $(impl $impls for $name {})+
    };
}

marker! {
    storage ComputeWrite = ComputeShaderStorageWrite, Write
}

marker! {
    storage ComputeRead = ComputeShaderStorageRead, Read
}

marker! {
    storage StorageWrite = AnyShaderStorageWrite, Write
}

marker! {
    storage StorageRead = AnyShaderStorageRead, Read
}

marker! {
    transfer TransferWrite = TransferWrite, Write
}

marker! {
    transfer TransferRead = TransferRead, Read
}

marker! {
    sampled ComputeSampledRead = ComputeShaderSampledRead
}

marker! {
    sampled FragmentSampledRead = FragmentShaderSampledRead
}

marker! {
    storage_image FragmentWrite = FragmentShaderStorageWrite, Write
}

marker! {
    attachment ColorWrite = ColorAttachmentWrite
}

marker! {
    attachment DepthWrite = DepthStencilAttachmentWrite
}

marker! {
    buffer UniformRead = AnyShaderUniformRead
}

marker! {
    buffer VertexRead = VertexBuffer
}

marker! {
    buffer IndexRead = IndexBuffer
}

marker! {
    buffer IndirectRead = IndirectBuffer
}
