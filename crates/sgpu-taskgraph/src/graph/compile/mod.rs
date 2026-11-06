pub(crate) mod alias;
pub(crate) mod schedule;
pub(crate) mod sync;
pub(crate) mod timeline;

use crate::{
    PassBufferId,
    graph::{ErasedNode, ResourcePool},
    node::ImageUse,
};
use sgpu::{AccessType, QueueType};

use alias::{Heap, assign_lane_waits, build_alias_plan};
use schedule::Schedule;
use sync::{SubmitPlan, build_submit_plan};
use timeline::build_timelines;

pub use timeline::Resource;

pub(crate) const QUEUES: [QueueType; 3] = [
    QueueType::Graphics,
    QueueType::Compute,
    QueueType::Transfer,
];

/// A node and the resources it accesses.
pub(crate) struct NodeDeclaration<'g, R, D> {
    pub(crate) label: &'static str,
    pub(crate) queue: QueueType,
    pub(crate) images: Vec<ImageUse>,
    pub(crate) buffers: Vec<(PassBufferId, AccessType)>,
    pub(crate) built: Box<dyn ErasedNode<R, D> + 'g>,
}

/// Why a graph failed to plan.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// The declared accesses form a dependency cycle; no execution order
    /// exists. Lists node indices and their labels.
    DependencyCycle {
        /// Nodes in, or downstream of, the cycle.
        nodes: Vec<usize>,
        /// Type names of those nodes, in the same order.
        labels: Vec<&'static str>,
    },
    /// One node accesses the same resource in two conflicting ways, which
    /// would require an ordering inside the node.
    SelfDependency {
        /// The node index.
        node: usize,
        /// The node's type name.
        label: &'static str,
        /// The resource it declared twice.
        resource: Resource,
        /// The first access it declared.
        first: AccessType,
        /// The access that could not be merged with the first.
        second: AccessType,
    },
    /// A resource is accessed after it was left in the `Present` state.
    PresentAfterAccess {
        /// The resource.
        resource: Resource,
    },
    /// A resource is presented without ever being written.
    PresentWithoutWrite {
        /// The resource.
        resource: Resource,
    },
    /// A transient resource's first access is not a write: its contents are
    /// undefined, so nothing may legally read it first.
    TransientNotWrittenFirst {
        /// The resource.
        resource: Resource,
    },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::DependencyCycle {
                nodes,
                labels,
            } => {
                write!(f, "dependency cycle between nodes:")?;
                for (node, label) in nodes.iter().zip(labels) {
                    write!(f, " [{node} {label}]")?;
                }
                Ok(())
            }
            GraphError::SelfDependency {
                node,
                label,
                resource,
                first,
                second,
            } => write!(f, "node {node} ({label}) accesses {resource:?} with conflicting accesses {first:?} and {second:?} in one phase; split it into phases or combine the accesses"),
            GraphError::PresentAfterAccess { resource } => write!(f, "{resource:?} is accessed after it was left in the Present state"),
            GraphError::PresentWithoutWrite { resource } => write!(f, "{resource:?} is presented without ever being written"),
            GraphError::TransientNotWrittenFirst { resource } => write!(f, "transient {resource:?} must start with a write; its contents are undefined before that"),
        }
    }
}

impl std::error::Error for GraphError {}

/// Plans a graph end to end: timelines, waves, aliasing, submits.
pub(crate) fn build<'g, R, D>(pool: &ResourcePool, nodes: &[NodeDeclaration<'g, R, D>]) -> Result<(SubmitPlan, Vec<Heap>), GraphError> {
    let mut timelines = build_timelines(pool, nodes)?;
    let schedule = Schedule::build(&mut timelines, nodes)?;

    let heaps = build_alias_plan(pool, &timelines)?;
    assign_lane_waits(&heaps, &mut timelines);

    Ok((build_submit_plan(&timelines, &schedule, nodes), heaps))
}

#[cfg(test)]
mod tests {
    use std::mem::ManuallyDrop;

    use sgpu::{ImageAspect, ImageDescription, ImageSubresources, ImageViewDescription, ImageViewType};

    use super::{sync::CompiledImageBarrier, *};
    use crate::{
        PassImageId,
        graph::InnerImage,
        node::{ExecutionContext, Node, NodeWriter},
        resource::ImageViewRef,
    };

    /// A node that declares nothing: the tests fill `NodeDeclaration` by hand.
    struct TestNode;

    impl Node<(), ()> for TestNode {
        type Config = ();

        fn declare(_: &(), _: &(), _: &mut NodeWriter) -> Self { TestNode }

        fn execute(&self, _: &(), _: &mut ExecutionContext<'_>) {}
    }

    fn declaration(images: Vec<ImageUse>) -> NodeDeclaration<'static, (), ()> {
        NodeDeclaration {
            label: "TestNode",
            queue: QueueType::Graphics,
            images,
            buffers: Vec::new(),
            built: Box::new(TestNode),
        }
    }

    /// A pool holding one graph-owned image. Planning touches no GPU state, and
    /// the pool is never dropped (which would wait on an uninitialized GPU).
    fn test_pool() -> (ManuallyDrop<ResourcePool>, PassImageId) {
        let mut pool = ManuallyDrop::new(ResourcePool::new());
        let image = pool.push_image(InnerImage::Persistent {
            image_desc: ImageDescription::default(),
            images: Vec::new(),
            resize: false,
        });

        (pool, image)
    }

    fn view_desc(mip_level: u32, level_count: u32) -> ImageViewDescription {
        ImageViewDescription {
            view_type: ImageViewType::Type2D,
            subresources: ImageSubresources {
                mip_level,
                level_count,
                ..ImageSubresources::default()
            },
        }
    }

    fn register(pool: &mut ResourcePool, image: PassImageId, desc: ImageViewDescription) -> ImageViewRef {
        let image = pool.check_image_raw(image);
        ImageViewRef::Registered(pool.register_view(image, desc))
    }

    /// Every planned image barrier of every submit, in submission order.
    fn barriers(plan: &SubmitPlan) -> Vec<CompiledImageBarrier> { plan.submits.iter().flatten().flat_map(|submit| submit.exec_batches.iter()).flat_map(|batch| batch.before_barriers.images.iter().cloned()).collect() }

    #[test]
    fn disjoint_views_share_one_group() {
        let (mut pool, image) = test_pool();
        let mip0 = register(&mut pool, image, view_desc(0, 1));
        let mip1 = register(&mut pool, image, view_desc(1, 1));

        let nodes = vec![
            declaration(vec![ImageUse {
                image,
                view: mip0,
                access: AccessType::ColorAttachmentWrite,
            }]),
            declaration(vec![ImageUse {
                image,
                view: mip1,
                access: AccessType::TransferWrite,
            }]),
        ];

        let timelines = build_timelines(&pool, &nodes).expect("disjoint views are not a self dependency");
        assert_eq!(timelines.len(), 1);
        assert_eq!(timelines[0].groups.len(), 1, "views of disjoint subresources belong in one group");
        assert_eq!(timelines[0].groups[0].members[0], vec![0, 1]);

        // Each barrier names one view and only the accesses covering it, so
        // neither barrier can see the other view's write.
        let (plan, _) = build(&pool, &nodes).expect("the plan builds");
        let planned = barriers(&plan);
        assert_eq!(planned.len(), 2, "one barrier per view instead of one whole-image barrier");

        for barrier in &planned {
            let expected = if barrier.view == mip0 {
                AccessType::ColorAttachmentWrite
            } else {
                AccessType::TransferWrite
            };
            assert_eq!(barrier.previous, vec![expected], "barrier {barrier:?} only sees its own view's source");
            assert_eq!(barrier.next, vec![expected], "barrier {barrier:?} only sees its own view's destination");
        }
    }

    #[test]
    fn overlapping_views_need_two_groups() {
        let (mut pool, image) = test_pool();
        let wide = register(&mut pool, image, view_desc(0, 2));
        let narrow = register(&mut pool, image, view_desc(1, 1));

        let nodes = vec![
            declaration(vec![ImageUse {
                image,
                view: wide,
                access: AccessType::ColorAttachmentWrite,
            }]),
            declaration(vec![ImageUse {
                image,
                view: narrow,
                access: AccessType::ComputeShaderSampledRead,
            }]),
        ];

        let timelines = build_timelines(&pool, &nodes).expect("two nodes never self-depend");
        assert_eq!(timelines[0].groups.len(), 2, "overlapping subresources must be ordered");
    }

    #[test]
    fn registered_view_reaches_the_barrier() {
        let (mut pool, image) = test_pool();
        let mip1 = register(&mut pool, image, view_desc(1, 1));

        let nodes = vec![
            declaration(vec![ImageUse {
                image,
                view: ImageViewRef::Default,
                access: AccessType::ColorAttachmentWrite,
            }]),
            declaration(vec![ImageUse {
                image,
                view: mip1,
                access: AccessType::ComputeShaderSampledRead,
            }]),
        ];

        let (plan, _) = build(&pool, &nodes).expect("the plan builds");
        let barrier = barriers(&plan).into_iter().find(|barrier| barrier.view == mip1).expect("the registered view reaches the barrier");

        assert_eq!(barrier.previous, vec![AccessType::ColorAttachmentWrite]);
        assert_eq!(barrier.next, vec![AccessType::ComputeShaderSampledRead]);
    }

    /// `AUTO` is resolved against the image's format when the view is created,
    /// which the plan cannot see, so it may never be assumed disjoint from a
    /// literal aspect.
    #[test]
    fn auto_aspect_overlaps_every_literal_aspect() {
        let (mut pool, image) = test_pool();
        let desc = |aspect| ImageViewDescription {
            subresources: ImageSubresources {
                aspect,
                ..ImageSubresources::default()
            },
            ..ImageViewDescription::default()
        };
        let auto = register(&mut pool, image, desc(ImageAspect::AUTO));
        let depth = register(&mut pool, image, desc(ImageAspect::DEPTH));

        let nodes = vec![
            declaration(vec![ImageUse {
                image,
                view: auto,
                access: AccessType::ColorAttachmentWrite,
            }]),
            declaration(vec![ImageUse {
                image,
                view: depth,
                access: AccessType::ComputeShaderSampledRead,
            }]),
        ];

        let timelines = build_timelines(&pool, &nodes).expect("two nodes never self-depend");
        assert_eq!(timelines[0].groups.len(), 2, "an unspecified aspect cannot be proven disjoint from a literal one");
    }
}
