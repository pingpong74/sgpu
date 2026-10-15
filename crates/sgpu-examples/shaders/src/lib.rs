// Major note:
// Since this requires a seperate tool chain,
// rust analyzer cannot offer its services

#![cfg_attr(target_arch = "spirv", no_std)]

pub mod cube;
pub mod post;
pub mod triangle;
