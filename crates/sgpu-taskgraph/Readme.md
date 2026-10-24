# Taskgraph for SGPU

One of the hardest and most error prone parts of vulkan is syncronization 
SGPU taskgraph automated syncronization within the same queue and between queues as well.
Each group of GPU work is marked by the trait `Node` which is declares how the resources are 
going to be used by this particular node. 
The task grap uses this information to perform aliasing and syncronization.
