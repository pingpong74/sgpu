use std::fmt;
use std::marker::PhantomData;

use sgpu::{Image, ImageDescription, ImageView, ImageViewDescription, destroy_image_view};
use smallvec::SmallVec;

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

/// Graph image view with its Access Type
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PassView<A: ImageAccess> {
    pub(crate) index: u32,
    _m: PhantomData<fn() -> A>,
}

impl<A: ImageAccess> fmt::Debug for PassView<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_struct("PassView").field("index", &self.index).finish() }
}

impl<A: ImageAccess> PassView<A> {
    pub(crate) fn new(index: u32) -> Self {
        PassView {
            index,
            _m: PhantomData,
        }
    }
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

#[derive(Default)]
pub(crate) struct ImageViewPool {
    specs: Vec<ImageViewSpecification>,
    retired: Vec<ImageView>,
}

impl ImageViewPool {
    pub(crate) fn register(&mut self, image_id: u32, desc: ImageViewDescription) -> u32 {
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

    pub(crate) fn refresh(&mut self, images: &[InnerImage]) {
        for spec in &mut self.specs {
            let image = images[spec.image_id as usize].image();
            if image == Image::default() {
                continue;
            }
            if spec.entries.iter().any(|e| e.image == image) {
                continue;
            }
            let evicted = (spec.entries.len() == MAX_VIEWS_PER_SPEC).then(|| spec.entries.remove(0).view);
            spec.entries.push(ViewEntry {
                image,
                view: image.create_view(&spec.image_view_desc),
            });
            if let Some(view) = evicted {
                self.retired.push(view);
            }
        }
        if self.retired.len() >= RETIRE_DRAIN_AT {
            self.drain();
        }
    }

    pub(crate) fn get(&self, index: u32, images: &[InnerImage]) -> ImageView {
        let spec = &self.specs[index as usize];
        let image = images[spec.image_id as usize].image();
        spec.entries.iter().find(|e| e.image == image).expect("image view not created (image unbound?)").view
    }

    // moves intems into retired queue
    pub(crate) fn destroy_slot(&mut self, image_id: u32) {
        for spec in &mut self.specs {
            if spec.image_id == image_id {
                self.retired.extend(spec.entries.drain(..).map(|entry| entry.view));
            }
        }
    }

    // destroys all the views
    pub(crate) fn destory_all(&mut self) {
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
    pub(crate) fn drain(&mut self) {
        if self.retired.is_empty() {
            return;
        }
        sgpu::wait_idle();
        for view in self.retired.drain(..) {
            destroy_image_view(view);
        }
    }
}
