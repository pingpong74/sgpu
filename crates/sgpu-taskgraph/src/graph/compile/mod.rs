pub(crate) mod alias;
pub(crate) mod schedule;
pub(crate) mod sync_2;
pub(crate) mod timeline;

use crate::{
    PassBufferId, PassImageId,
    graph::{ErasedNode, ResourcePool},
};
use sgpu::{AccessType, QueueType};

use alias::{Heap, assign_lane_waits, build_alias_plan};
use schedule::Schedule;
use sync_2::{SubmitPlan, build_submit_plan};
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
    pub(crate) images: Vec<(PassImageId, AccessType)>,
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
