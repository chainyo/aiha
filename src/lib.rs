#![deny(missing_docs)]
//! # AIHA - AI Hardware Advisor
//! 
//! Introducing the **AI Hardware Advisor (AIHA)** – Unlock the Power of Optimal Hardware for **Your AI Project**!
//! 
//! Looking to embark on an AI journey? Look no further! **AI Hardware Advisor (AIHA)** is the ultimate tool that empowers
//! you to make intelligent decisions when selecting hardware for your AI endeavors.
//! 
//! With **AIHA**, the guessing game is over. Say goodbye to uncertainty and welcome a world of precise resource allocation
//! for inference and training any model on the esteemed Hugging Face Hub.
//!
pub mod hardware;
pub use hardware::{ scan_hardware, Hardware, NvidiaDevice };
