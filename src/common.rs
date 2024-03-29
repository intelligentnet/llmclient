use std::pin::Pin;
use serde::ser::StdError;
use serde_derive::Deserialize;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub enum LlmType  {
    GEMINI,
    GPT,
    CLAUDE,
    MISTRAL,
    GEMINI_ERROR,
    GPT_ERROR,
    CLAUDE_ERROR,
    MISTRAL_ERROR,
}

pub type Triple = (usize, usize, usize);

impl std::fmt::Display for LlmType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LlmType::GEMINI => write!(f, "GEMINI"),
            LlmType::GPT => write!(f, "GPT"),
            LlmType::CLAUDE => write!(f, "CLAUDE"),
            LlmType::MISTRAL => write!(f, "MISTRAL"),
            LlmType::GEMINI_ERROR => write!(f, "GEMINI_ERROR"),
            LlmType::GPT_ERROR => write!(f, "GPT_ERROR"),
            LlmType::CLAUDE_ERROR => write!(f, "CLAUDE_ERROR"),
            LlmType::MISTRAL_ERROR => write!(f, "MISTRAL_ERROR"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmReturn {
    pub llm_type: LlmType,
    pub text: String,
    pub finish_reason: String,
    pub usage: Triple,
    pub timing: f64,
    pub citations: Option<String>,
    pub safety_ratings: Option<Vec<String>>,
}

impl LlmReturn {
    pub fn new(llm_type: LlmType, text: String, finish_reason: String, usage: Triple, timing: f64, citations: Option<String>, safety_ratings: Option<Vec<String>>) -> Self {
        LlmReturn { llm_type, text, finish_reason, usage, timing, citations, safety_ratings }
    }
}

#[allow(clippy::print_in_format_impl)]
impl std::fmt::Display for LlmReturn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        println!("---------- {} ----------", self.llm_type);
        let _ = writeln!(f, "{}", self.text);
        if !self.finish_reason.is_empty() && self.finish_reason != "STOP" {
            println!("Finish Reason: {}", self.finish_reason);
        }
        println!("Tokens: Input: {} + Output: {} -> Total: {}",
                 self.usage.0, self.usage.1, self.usage.2);
        println!("Timing: {:.4} secs", self.timing);
        if let Some(ref citations) = self.citations {
            println!("Citations:\n{}", citations);
        }
        if let Some(ref safety_ratings) = self.safety_ratings {
            println!("Safety Settings: {:?}", safety_ratings);
        }

        Ok(())
    }
}

pub trait LlmCompletion {
    /*
    /// Create and call llm by supplying data and common parameters
    fn call(system: &str, user: &Vec<&str>, temperature: f32, is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>>;
    */

    /// Supply single role and single part text
    fn add_text(&mut self, role: &str, content: &str);

    /// Supply single role with multi-string for iparts with single content
    fn add_many_text(&mut self, role: &str, prompt: &[String]);

    /// Supply simple, 'system' content
    fn add_system(&mut self, system_prompt: &str);

    /// Supply multi-parts and single 'system' content
    fn add_multi_part_system(&mut self, system_prompts: &[String]);

    /// Supply multi-context 'system' content
    fn add_systems(&mut self, system_prompts: &[String]);

    /// Supply multi-String content with user and llm alternating
    fn dialogue(&mut self, prompts: &[String], has_system: bool);

    /// Truncate messages
    fn truncate_messages(&mut self, len: usize);

    /// Return String of Object
    fn debug(&self) -> String;

    // Set content in precreated completion
    //fn set_content(&mut self, content: Vec<Box<dyn LlmMessage>>);

    /// Default call to LLM so trait can be used for simple calls
    fn call_llm(&'static self) -> Pin<Box<(dyn futures::Future<Output = Result<LlmReturn, Box<(dyn StdError + Send + 'static)>>> + Send + 'static)>>;
}

pub trait LlmMessage {
    /// Supply single role and single part text
    fn text(role: &str, content: &str) -> Self
        where Self: Sized;

    /// Supply single role with multi-string for iparts with single content
    fn many_text(role: &str, prompt: &[String]) -> Self
        where Self: Sized;

    /// Supply simple, 'system' content
    fn system(system_prompt: &str) -> Vec<Self>
        where Self: Sized;

    /// Supply multi-parts and single 'system' content
    fn multi_part_system(system_prompts: &[String]) -> Vec<Self>
        where Self: Sized;

    /// Supply multi-context 'system' content
    fn systems(system_prompts: &[String]) -> Vec<Self>
        where Self: Sized;

    /// Supply multi-String content with user and model alternating
    fn dialogue(prompts: &[String], has_system: bool) -> Vec<Self>
        where Self: Sized;

    /// Return String of Object
    fn debug(&self) -> String;
}

// Lowest common denominator error message!
#[derive(Debug, Deserialize)]
pub struct LlmError {
    pub error: LlmErrorMessage
}

#[derive(Debug, Deserialize)]
pub struct LlmErrorMessage {
    pub message: String
}

impl std::fmt::Display for LlmErrorMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "message: {}", self.message)
    }
}
