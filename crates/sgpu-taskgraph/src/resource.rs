use std::fmt;
use std::marker::PhantomData;

use sgpu::{Image, ImageDescription};

use crate::access::{BufferAccess, ImageAccess};

/// Cheap Index to Graph owned resources
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Id<T> {
    pub(crate) index: u32,
    pub(crate) graph: u32,
    _m: PhantomData<fn() -> T>,
}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("Id").field("index", &self.index).field("graph", &self.graph).finish() }
}

impl<T> Id<T> {
    pub(crate) fn new(index: u32, graph: u32) -> Self {
        Id {
            index,
            graph,
            _m: PhantomData,
        }
    }

    /// Panics unless the id was minted by this graph and is in range
    pub(crate) fn check(self, graph: u32, len: usize) -> u32 {
        assert!(self.graph == graph && (self.index as usize) < len, "id {self:?} does not belong to this graph");
        self.index
    }
}

/// Graph Image with its Access Type
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PassImage<A: ImageAccess> {
    pub(crate) index: u32,
    _m: PhantomData<fn() -> A>,
}

impl<A: ImageAccess> fmt::Debug for PassImage<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("PassImage").field("index", &self.index).finish() }
}

impl<A: ImageAccess> PassImage<A> {
    pub(crate) fn new(index: u32) -> Self {
        PassImage {
            index,
            _m: PhantomData,
        }
    }
}

/// Graph Buffer with its Access Type
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PassBuffer<A: BufferAccess> {
    pub(crate) index: u32,
    _m: PhantomData<fn() -> A>,
}

impl<A: BufferAccess> fmt::Debug for PassBuffer<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("PassBuffer").field("index", &self.index).finish() }
}

impl<A: BufferAccess> PassBuffer<A> {
    pub(crate) fn new(index: u32) -> Self {
        PassBuffer {
            index,
            _m: PhantomData,
        }
    }
}

/// Who owns an image registered with the graph
pub(crate) enum InnerImage {
    /// User owned image, graph just borrows the handle
    External(Image),
    /// Created and owned by the graph
    Persistent {
        image_desc: ImageDescription,
        image: Image,
        resizable: bool,
    },
}

impl InnerImage {
    pub(crate) fn image(&self) -> Image {
        match self {
            Self::External(image) | Self::Persistent { image, .. } => *image,
        }
    }
}
