use serde_derive::Deserialize;
use reqwest::Client;
use reqwest::header::{HeaderMap, HeaderValue};
use crate::gemini::GeminiCompletion;
use crate::gpt::GptCompletion;
use crate::mistral::MistralCompletion;
use crate::claude::ClaudeCompletion;
use crate::groq::GroqCompletion;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub enum LlmType  {
    GEMINI,
    GPT,
    CLAUDE,
    MISTRAL,
    GROQ,
    GEMINI_ERROR,
    GPT_ERROR,
    CLAUDE_ERROR,
    MISTRAL_ERROR,
    GROQ_ERROR,
}

pub type Triple = (usize, usize, usize);

impl std::fmt::Display for LlmType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LlmType::GEMINI => write!(f, "GEMINI"),
            LlmType::GPT => write!(f, "GPT"),
            LlmType::CLAUDE => write!(f, "CLAUDE"),
            LlmType::MISTRAL => write!(f, "MISTRAL"),
            LlmType::GROQ => write!(f, "GROQ"),
            LlmType::GEMINI_ERROR => write!(f, "GEMINI_ERROR"),
            LlmType::GPT_ERROR => write!(f, "GPT_ERROR"),
            LlmType::CLAUDE_ERROR => write!(f, "CLAUDE_ERROR"),
            LlmType::MISTRAL_ERROR => write!(f, "MISTRAL_ERROR"),
            LlmType::GROQ_ERROR => write!(f, "GROQ_ERROR"),
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
    /// Set temperature
    fn set_temperature(&mut self, temperature: f32);

    /// If applicable set output to be json. Hint in prompt still necessary.
    fn set_json(&mut self, _is_json: bool) {
        // not applicable for all models
    }

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
 
    fn call(system: &str, user: &[String], temperature: f32, _is_json: bool, is_chat: bool) -> impl std::future::Future<Output = Result<LlmReturn, Box<dyn std::error::Error + Send>>> + Send;
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

/// Call named LLM with common parameters supplied
pub async fn call_llm(llm: &str, system: &str, user: &[String], temperature: f32, is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    match llm {
        "google" | "gemini" => {
            GeminiCompletion::call(system, user, temperature, is_json, is_chat).await
        },
        "openai" | "gpt" => {
            GptCompletion::call(system, user, temperature, is_json, is_chat).await
        },
        "mistral" => {
            MistralCompletion::call(system, user, temperature, is_json, is_chat).await
        },
        "anthropic" | "claude" => {
            ClaudeCompletion::call(system, user, temperature, is_json, is_chat).await
        },
        _ => {
            GroqCompletion::call(system, user, temperature, is_json, is_chat).await
        },
    }
}

/// Call default (see LLM_TO_USE env var) LLM with common parameters supplied
pub async fn call(system: &str, user: &[String], temperature: f32, is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let llm: &str = &std::env::var("LLM_TO_USE").map_err(|_| "groq".to_string()).unwrap();

    call_llm(llm, system, user, temperature, is_json, is_chat).await
}

/// Call single shot default LLM with default values for parameters supplied
pub async fn single_call(system: &str, user: &[String]) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {

    call(system, user, 0.2, false, false).await
}

/// Call single shot default LLM with default values for parameters supplied
/// Should return JSON
pub async fn single_call_json(system: &str, user: &[String]) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let system = &format!("Return valid JSON only. {system}");

    call(system, user, 0.2, true, false).await
}

/// Call chat default LLM with default values for parameters supplied
pub async fn chat_call(system: &str, user: &[String]) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {

    call(system, user, 0.2, false, true).await
}

/// Call chat default LLM with default values for parameters supplied
/// Should return JSON
pub async fn chat_call_json(system: &str, user: &[String]) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let system = &format!("Return valid JSON only. {system}");

    call(system, user, 0.2, true, true).await
}

/// Call single shot default LLM with temperature supplied
pub async fn single_call_temperature(system: &str, user: &[String], temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {

    call(system, user, temperature, false, false).await
}

/// Call single shot default LLM with temperature supplied
/// Should return JSON
pub async fn single_call_json_temperature(system: &str, user: &[String], temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let system = &format!("Return valid JSON only. {system}");

    call(system, user, temperature, true, false).await
}

/// Call chat default LLM with temperature supplied
pub async fn chat_call_temperature(system: &str, user: &[String], temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {

    call(system, user, temperature, false, true).await
}

/// Call chat default LLM with temperature supplied
/// Should return JSON
pub async fn chat_call_json_temperature(system: &str, user: &[String], temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let system = &format!("Return valid JSON only. {system}");

    call(system, user, temperature, true, true).await
}

/// Call single shot named LLM with default values for parameters supplied
pub async fn single_call_llm(llm: &str, system: &str, user: &[String]) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {

    call_llm(llm, system, user, 0.2, false, false).await
}

/// Call single shot named LLM with default values for parameters supplied
/// Should return JSON
pub async fn single_call_json_llm(llm: &str, system: &str, user: &[String]) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let system = &format!("Return valid JSON only. {system}");

    call_llm(llm, system, user, 0.2, true, false).await
}

/// Call chat named LLM with default values for parameters supplied
pub async fn chat_call_llm(llm: &str, system: &str, user: &[String]) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {

    call_llm(llm, system, user, 0.2, false, true).await
}

/// Call chat named LLM with default values for parameters supplied
/// Should return JSON
pub async fn chat_call_json_llm(llm: &str, system: &str, user: &[String]) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let system = &format!("Return valid JSON only. {system}");

    call_llm(llm, system, user, 0.2, true, true).await
}

/// Call single shot named LLM with temperature supplied
pub async fn single_call_temperature_llm(llm: &str, system: &str, user: &[String], temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {

    call_llm(llm, system, user, temperature, false, false).await
}

/// Call single shot named LLM with temperature supplied
/// Should return JSON
pub async fn single_call_json_temperature_llm(llm: &str, system: &str, user: &[String], temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let system = &format!("Return valid JSON only. {system}");

    call_llm(llm, system, user, temperature, true, false).await
}

/// Call chat named LLM with temperature supplied
pub async fn chat_call_temperature_llm(llm: &str, system: &str, user: &[String], temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {

    call_llm(llm, system, user, temperature, false, true).await
}

/// Call chat named LLM with temperature supplied
/// Should return JSON
pub async fn chat_call_json_temperature_llm(llm: &str, system: &str, user: &[String], temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let system = &format!("Return valid JSON only. {system}");

    call_llm(llm, system, user, temperature, true, true).await
}

/// Common HTTP client with header setup
pub async fn get_client(mut headers: HeaderMap) -> Result<Client, Box<dyn std::error::Error + Send>> {
    // We would like json
    headers.insert(
        "Content-Type",
        HeaderValue::from_str("appication/json; charset=utf-8")
            .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?,
    );

    // Create client
    let client: Client = Client::builder()
        .user_agent("TargetR")
        .timeout(std::time::Duration::new(120, 0))
        //.gzip(true)
        .default_headers(headers)
        .build()
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?;

    Ok(client)
}
