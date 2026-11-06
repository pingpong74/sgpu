use std::fmt;
use std::marker::PhantomData;
use std::num::NonZero;

use crate::access::{BufferAccess, ImageAccess};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PassImageId(pub(crate) u32, u32);

impl PassImageId {
    pub(crate) fn new(index: u32, graph: u32) -> Self { Self(index, graph) }

    pub(crate) fn check_graph(&self, graph_id: u32) -> u32 {
        assert!(self.1 == graph_id, "id {self:?} does not belong to this graph");
        self.0
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PassBufferId(pub(crate) u32, u32);

impl PassBufferId {
    pub(crate) fn new(index: u32, graph: u32) -> Self { Self(index, graph) }

    pub(crate) fn check_graph(&self, graph_id: u32) -> u32 {
        assert!(self.1 == graph_id, "id {self:?} does not belong to this graph");
        self.0
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

/// Graph image view with its Access Type
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PassImageView<A: ImageAccess> {
    index: Option<NonZero<u32>>,
    pub(crate) image: u32,
    _m: PhantomData<fn() -> A>,
}

impl<A: ImageAccess> fmt::Debug for PassImageView<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("PassView").field("index", &self.index).finish() }
}

impl<A: ImageAccess> PassImageView<A> {
    pub(crate) fn new(index: u32, image: u32) -> Self {
        PassImageView {
            index: Some(NonZero::new(index + 1).expect("Pass Image View recieved wrong index")),
            image,
            _m: PhantomData,
        }
    }

    pub(crate) fn default_view(image: u32) -> Self {
        PassImageView {
            index: None,
            image,
            _m: PhantomData,
        }
    }

    pub(crate) fn get_index(&self) -> Option<u32> { self.index.map(|i| i.get() - 1) }
}

/// How an image access names its view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ImageViewRef {
    /// The image's default view.
    Default,
    /// A view registered through `NodeWriter::view`.
    Registered(u32),
}
