use std::pin::Pin;
use serde::ser::StdError;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use serde_derive::{Deserialize, Serialize};
use crate::common::*;
use crate::gpt::GptMessage as ClaudeMessage;

// Input structures
// Chat
#[derive(Debug, Serialize, Clone)]
pub struct ClaudeCompletion {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<ClaudeMessage>,
    pub temperature: f32,
    pub max_tokens: usize,
    //pub stream: bool,     // Not for now
    //pub top_p: u32,
    //pub top_k: u32,
}

impl ClaudeCompletion {
    /// Create chat completion
    pub fn new(messages: Vec<ClaudeMessage>, temperature: f32, _is_json: bool) -> Self {
        let model: String = env::var("CLAUDE_MODEL").expect("CLAUDE_MODEL not found in enviroment variables");

        ClaudeCompletion {
            model,
            system: None,
            messages,
            temperature,
            max_tokens: 4096,

        }
    }

    /// Create and call llm by supplying data and common parameters
    pub async fn call(system: &str, user: &[String], temperature: f32, _is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
        let model: String = env::var("CLAUDE_MODEL").expect("CLAUDE_MODEL not found in enviroment variables");
        let mut messages = Vec::new();

        user.iter()
            .enumerate()
            .for_each(|(i, c)| {
                let role = if !is_chat || i % 2 == 0 { "user" } else { "assistant" };

                messages.push(ClaudeMessage { role: role.into(), content: c.to_string() });
            });

        let completion = ClaudeCompletion {
            model,
            system: if system.is_empty() { None } else { Some(system.to_string()) },
            messages,
            temperature,
            max_tokens: 4096
        };

        call_claude_completion(&completion).await
    }

    pub fn set_model(&mut self, model: &str) {
        self.model = model.into();
    }

    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens;
    }

    /// Add a single new message
    pub fn add_message(&mut self, message: &ClaudeMessage) {
        self.messages.push(message.clone());
    }

    /// Add many new messages
    pub fn add_messages(&mut self, messages: &[ClaudeMessage]) {
        messages.iter().for_each(|m| self.messages.push(m.clone()));
    }
}

impl Default for ClaudeCompletion {
    /// Create default chat completion
    fn default() -> Self {
        let model: String = env::var("CLAUDE_MODEL").expect("CLAUDE_MODEL not found in enviroment variables");

        ClaudeCompletion {
            model,
            system: None,
            messages: Vec::new(),
            temperature: 0.2,
            max_tokens: 4096
        }
    }
}

impl LlmCompletion for ClaudeCompletion {
    /// Add single role and single part text
    fn add_text(&mut self, role: &str, text: &str) {
        self.messages.push(ClaudeMessage::text(role, text));
    }

    /// Add single role with multiple strings for parts as single large content
    fn add_many_text(&mut self, role: &str, texts: &[String]) {
        self.messages.push(ClaudeMessage::many_text(role, texts));
    }

    /// Supply simple, 'system' content
    fn add_system(&mut self, system_prompt: &str) {
        self.messages.append(&mut ClaudeMessage::system(system_prompt));
    }

    /// Supply multi-parts and single 'system' content
    fn add_multi_part_system(&mut self, system_prompts: &[String]) {
        self.messages.append(&mut ClaudeMessage::multi_part_system(system_prompts));
    }

    /// Supply multi-context 'system' content
    fn add_systems(&mut self, system_prompts: &[String]) {
        self.messages.append(&mut ClaudeMessage::systems(system_prompts));
    }

    /// Supply multi-String content with user and llm alternating
    fn dialogue(&mut self, prompts: &[String], has_system: bool) {
        self.messages = ClaudeMessage::dialogue(prompts, has_system);
    }
    
    /// Truncate messages
    fn truncate_messages(&mut self, len: usize) {
        self.messages.truncate(len);
    }

    /// Return String of Object
    fn debug(&self) -> String where Self: std::fmt::Debug {
        format!("{:?}", self)
    }

    /// Default call to LLM so trait can be used for simple calls
    fn call_llm(&'static self) -> Pin<Box<(dyn futures::Future<Output = Result<LlmReturn, Box<(dyn StdError + Send + 'static)>>> + Send + 'static)>> {
        Box::pin(call_claude_completion(self))
    }
}

// Output structures
// Chat
#[derive(Debug, Deserialize)]
pub struct ClaudeResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Option<Vec<Content>>,
    pub model: String,
    pub stop_reason: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct Content {
    pub r#type: String,
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

impl Usage {
    pub fn new() -> Self {
        Usage { input_tokens: 0, output_tokens: 0 }
    }

    pub fn to_triple(&self) -> (usize, usize, usize) {
        (self.input_tokens, self.output_tokens, self.input_tokens + self.output_tokens)
    }
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} + {} = {}", self.input_tokens, self.output_tokens, self.input_tokens + self.output_tokens)
    }
}

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
}

/// Call Claude with some messages
pub async fn call_claude(messages: Vec<ClaudeMessage>) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_claude_all(messages, 0.2, 4096).await
}

/// Call Claude with some messages and temperature
pub async fn call_claude_temperature(messages: Vec<ClaudeMessage>, temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_claude_all(messages, temperature, 4096).await
}

/// Call Claude with some messages and max_tokens
pub async fn call_claude_max_tokens(messages: Vec<ClaudeMessage>, max_tokens: usize) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_claude_all(messages, 0.2, max_tokens).await
}

/// Call Claude with some messages, temperature and max_tokens
pub async fn call_claude_all(messages: Vec<ClaudeMessage>, temperature: f32, max_tokens: usize) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    // Model/version of lln
    let model: String =
        env::var("CLAUDE_MODEL").expect("CLAUDE_MODEL not found in enviroment variables");
    let smess = extract_role("system", &messages);
    let umess = extract_role("user", &messages);

    // Create chat completion
    let claude_completion: ClaudeCompletion = ClaudeCompletion {
        model,
        system: if smess.is_empty() { None } else { Some(smess) },
        messages: vec![ClaudeMessage { role: "user".into(), content: umess }],
        temperature,
        max_tokens,
    };

    call_claude_completion(&claude_completion).await
}

/// Call Claude with pre-assembled completion
pub async fn call_claude_completion(claude_completion: &ClaudeCompletion) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let start = std::time::Instant::now();
    // url to call anthropic
    let url: String =
        env::var("CLAUDE_URL").expect("CLAUDE_URL not found in environment variables");

    let client = get_claude_client().await?;

    // Extract API Response
    let res = client
        .post(url)
        .json(&claude_completion)
        .send()
        .await;
    //let res: ClaudeResponse = res
    let res = res
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?
        //.json()
        .text()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?;
     
    let timing = start.elapsed().as_secs() as f64 + start.elapsed().subsec_millis() as f64 / 1000.0;

    if res.contains("\"error\":") {
        let res: LlmError = serde_json::from_str(&res).unwrap();

        Ok(LlmReturn::new(LlmType::CLAUDE_ERROR, res.error.to_string(), res.error.to_string(), (0, 0, 0), timing, None, None))
    } else {
        let res: ClaudeResponse = serde_json::from_str::<ClaudeResponse>(&res).unwrap();

        // Send Response
        let text =
            match res.content {
                Some(content) => {
                    let text = content.iter().map(|s| s.text.lines().filter(|l| !l.starts_with("```")).fold(String::new(), |s, l| s + l + "\n")).collect();

                    text
                },
                None => {
                    //Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "No content found")))
                    "No content found".to_string()
                }
            };
        let finish_reason = if res.stop_reason == "end_turn" { "STOP".to_string() } else { res.stop_reason };
        let usage: Triple = res.usage.to_triple();
        let timing = start.elapsed().as_secs() as f64 + start.elapsed().subsec_millis() as f64 / 1000.0;

        Ok(LlmReturn::new(LlmType::CLAUDE, text, finish_reason, usage, timing, None, None))
    }
}

fn extract_role(role: &str, messages: &[ClaudeMessage]) -> String {
    messages.iter()
        .filter(|m| role == m.role)
        .fold(String::new(), |mut s, i| {
            if !s.is_empty() {
                s.push('\n');
            }
            s.push_str(&i.content);

            s
        })
}

async fn get_claude_client() -> Result<Client, Box<dyn std::error::Error + Send>> {
    // Extract API Key information
    let api_key: String =
        env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not found in environment variables");
    // Date when version was available
    let version: String =
        env::var("CLAUDE_VERSION").expect("CLAUDE_VERSION not found in environment variables");

    // Create headers
    let mut headers: HeaderMap = HeaderMap::new();

    // Create api key header
    headers.insert(
        "x-api-key",
        HeaderValue::from_str(&api_key)
            .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?,
    );
    // Create api key header
    headers.insert(
        "anthropic-version",
        HeaderValue::from_str(&version)
            .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?,
    );

    get_client(headers).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    async fn claude(content: Vec<ClaudeMessage>) {
        match call_claude(content).await {
            Ok(ret) => { println!("{ret}"); assert!(true) },
            Err(e) => { println!("{e}"); assert!(false) },
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_call_claude_basic() {
        let messages: Vec<ClaudeMessage> = vec![ClaudeMessage { role: "user".into(), content: "What is the meaining of life?".into() }];

        claude(messages).await;
    }
    #[tokio::test]
    #[serial]
    async fn test_call_claude_citation() {
        let messages = 
            vec![ClaudeMessage::text("user", "Give citations for the General theory of Relativity.")];
        claude(messages).await;
    }
    #[tokio::test]
    //#[serial]
    async fn test_call_claude_poem() {
        let messages = 
            vec![ClaudeMessage::text("user", "Write a creative poem about the interplay of artificial intelligence and the human spirit and provide citations")];
        claude(messages).await;
    }
    #[tokio::test]
    //#[serial]
    async fn test_call_claude_logic() {
        let messages = 
            vec![ClaudeMessage::text("user", "How many brains does an octopus have, when they have been injured and lost a leg?")];
        claude(messages).await;
    }
}
