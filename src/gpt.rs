use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use serde_derive::{Deserialize, Serialize};
use crate::common::*;

// Input structures
// Chat

// Main chat object
#[derive(Debug, Serialize, Clone)]
pub struct GptCompletion {
    pub model: String,
    pub messages: Vec<GptMessage>,
    pub response_format: ResponseFormat,
    pub temperature: f32,
}

impl GptCompletion {
    /// Create chat completion
    pub fn new(messages: Vec<GptMessage>, temperature: f32, is_json: bool) -> Self {
        let model: String = env::var("GPT_MODEL").expect("GPT_MODEL not found in enviroment variables");

        GptCompletion {
            model,
            messages,
            temperature,
            response_format: ResponseFormat::new(is_json)
        }
    }

    pub fn set_model(&mut self, model: &str) {
        self.model = model.into();
    }

    pub fn set_response_format(&mut self, response_format: &ResponseFormat) {
        self.response_format = response_format.clone();
    }

    /// Add a single new message
    pub fn add_message(&mut self, message: &GptMessage) {
        self.messages.push(message.clone());
    }

    /// Add many new messages
    pub fn add_messages(&mut self, messages: &[GptMessage]) {
        messages.iter().for_each(|m| self.messages.push(m.clone()));
    }
}

impl Default for GptCompletion {
    /// Create default chat completion
    fn default() -> Self {
        let model: String = env::var("GPT_MODEL").expect("GPT_MODEL not found in enviroment variables");

        GptCompletion {
            model,
            messages: Vec::new(),
            temperature: 0.2,
            response_format: ResponseFormat::new(false)
        }
    }
}

impl LlmCompletion for GptCompletion {
    /// Set temperature
    fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    /// Set output to be json. Hint in prompt still necessary.
    fn set_json(&mut self, is_json: bool) {
        self.response_format = ResponseFormat::new(is_json);
    }

    /// Add single role and single part text
    fn add_text(&mut self, role: &str, text: &str) {
        self.messages.push(GptMessage::text(role, text));
    }

    /// Add single role with multiple strings for parts as single large content
    fn add_many_text(&mut self, role: &str, texts: &[String]) {
        self.messages.push(GptMessage::many_text(role, texts));
    }

    /// Supply simple, 'system' content
    fn add_system(&mut self, system_prompt: &str) {
        self.messages.append(&mut GptMessage::system(system_prompt));
    }

    /// Supply multi-parts and single 'system' content
    fn add_multi_part_system(&mut self, system_prompts: &[String]) {
        self.messages.append(&mut GptMessage::multi_part_system(system_prompts));
    }

    /// Supply multi-context 'system' content
    fn add_systems(&mut self, system_prompts: &[String]) {
        self.messages.append(&mut GptMessage::systems(system_prompts));
    }

    /// Supply multi-String content with user and llm alternating
    fn dialogue(&mut self, prompts: &[String], has_system: bool) {
        self.messages = GptMessage::dialogue(prompts, has_system);
    }
    
    /// Truncate messages
    fn truncate_messages(&mut self, len: usize) {
        self.messages.truncate(len);
    }

    /// Return String of Object
    fn debug(&self) -> String where Self: std::fmt::Debug {
        format!("{:?}", self)
    }

    // Set content in precreated completion
    //fn set_content(&mut self, content: Vec<Box<dyn LlmMessage>>) {
    //    self.messages = content;
    //}

    /*
    /// Create and Call LLM
    fn create_call_llm(system: &Vec<&str>, user: &Vec<&str>, temperature: f32, is_json: bool, is_chat: bool) -> Pin<Box<(dyn futures::Future<Output = Result<LlmReturn, Box<(dyn StdError + Send + 'static)>>> + Send + 'static)>> {

        Box::pin(call_gpt_completion(llm))
    }
    */
    /// Create and call llm by supplying data and common parameters
    async fn call(system: &str, user: &[String], temperature: f32, is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
        let model: String = env::var("GPT_MODEL").expect("GPT_MODEL not found in enviroment variables");
        let mut messages = Vec::new();

        if !system.is_empty() {
            messages.push(GptMessage { role: "system".into(), content: system.into() });
        }

        user.iter()
            .enumerate()
            .for_each(|(i, c)| {
                let role = if !is_chat || i % 2 == 0 { "user" } else { "assistant" };

                messages.push(GptMessage { role: role.into(), content: c.to_string() });
            });

        let completion = GptCompletion {
            model,
            messages,
            temperature,
            response_format: ResponseFormat::new(is_json)
        };

        call_gpt_completion(&completion).await
    }

}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseFormat {
    pub r#type: String,
}

impl ResponseFormat {
    pub fn new(is_json: bool) -> Self {
        ResponseFormat { r#type: 
            if is_json {
                "json_object".to_string()
            } else {
                "text".to_string()
            }
        }
    }
}

/// Main Message Object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptMessage {
    pub role: String,
    pub content: String,
}

impl LlmMessage for GptMessage {
    /// Supply single role and single part text
    fn text(role: &str, content: &str) -> Self {
        Self { role: role.into(), content: content.into() }
    }

    /// Supply single role with multi-string for iparts with single content
    fn many_text(role: &str, prompt: &[String]) -> Self {
        let prompt: String = 
            prompt.iter()
                .fold(String::new(), |mut s, p| {
                    s.push_str(if s.is_empty() { "" } else { "\n" });
                    s.push_str(p);

                    s
                });

        Self { role: role.into(), content: prompt }
    }

    /// Supply simple, 'system' content
    fn system(system_prompt: &str) -> Vec<Self> {
        vec![Self::text("system", system_prompt)]
    }

    /// Supply multi-parts and single 'system' content
    fn multi_part_system(system_prompts: &[String]) -> Vec<Self> {
        vec![Self::many_text("system", system_prompts)]
    }

    /// Supply multi-context 'system' content
    fn systems(system_prompts: &[String]) -> Vec<Self> {
        system_prompts.iter()
            .map(|sp| Self::text("system", sp))
            .collect()
    }

    /// Supply multi-String content with user and model alternating
    fn dialogue(prompts: &[String], has_system: bool) -> Vec<Self> {
        prompts.iter()
            .enumerate()
            .map(|(i, p)| {
                let role = if i % 2 == 0 {
                    if i == 0 && has_system {
                        "system"
                    } else {
                        "user"
                    }
                } else {
                    "assistant"
                };

                Self::text(role, p)
            })
            .collect()
    }

    /// Return String of Object
    fn debug(&self) -> String where Self: std::fmt::Debug {
        format!("{:?}", self)
    }
}

// Output structures
// Chat
#[derive(Debug, Deserialize)]
pub struct GptResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub usage: Usage,
    pub choices: Option<Vec<GptChoice>>,
}

#[derive(Debug, Deserialize)]
pub struct GptChoice {
    pub message: GptMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<String>,
    pub finish_reason: String,
    pub index: usize
}

#[derive(Debug, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new() -> Self {
        Usage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    }

    pub fn to_triple(&self) -> (usize, usize, usize) {
        (self.prompt_tokens, self.completion_tokens, self.total_tokens)
    }
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} + {} = {}", self.prompt_tokens, self.completion_tokens, self.total_tokens)
    }
}

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
}

/// Call GPT with some messages
pub async fn call_gpt(messages: Vec<GptMessage>) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_gpt_all(messages, 0.2, false).await
}

/// Call GPT with some messages and option for Json
pub async fn call_gpt_json(messages: Vec<GptMessage>, is_json: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_gpt_all(messages, 0.2, is_json).await
}

/// Call GPT with some messages and temperature
pub async fn call_gpt_temperature(messages: Vec<GptMessage>, temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_gpt_all(messages, temperature, false).await
}

/// Call GPT with some messages, option for Json and temperature
pub async fn call_gpt_all(messages: Vec<GptMessage>, temperature: f32, is_json: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    // Create chat completion
    let gpt_completion = GptCompletion::new(messages, temperature, is_json);

    call_gpt_completion(&gpt_completion).await
}

/// Call Claude with pre-assembled completion
pub async fn call_gpt_completion(gpt_completion: &GptCompletion) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let start = std::time::Instant::now();
    // Confirm endpoint
    let url: String = env::var("GPT_CHAT_URL").expect("GPT_CHAT_URL not found in enviroment variables");

    let client = get_gpt_client().await?;

    // Extract API Response
    let res = client
        .post(url)
        .json(&gpt_completion)
        .send()
        .await;
    //let res: GptResponse = res
    let res = res
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?
        //.json()
        .text()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?;

    let timing = start.elapsed().as_secs() as f64 + start.elapsed().subsec_millis() as f64 / 1000.0;

    if res.contains("\"error\":") {
        let res: LlmError = serde_json::from_str(&res).unwrap();

        Ok(LlmReturn::new(LlmType::GPT_ERROR, res.error.to_string(), res.error.to_string(), (0, 0, 0), timing, None, None))
    } else {
        let res: GptResponse = serde_json::from_str::<GptResponse>(&res).unwrap();

        // Send Response
        let text: String =
            match res.choices {
                Some(ref choices) if !choices.is_empty() => {
                    // For now they only return one choice!
                    let text = choices[0].message.content.clone();
                    let text = text.lines().filter(|l| !l.starts_with("```")).fold(String::new(), |s, l| s + l + "\n");

                    text
                },
                Some(_) | None => {
                    "None".into()
                }
            };
        let finish_reason: String = 
            match res.choices {
                Some(ref choices) if !choices.is_empty() => {
                    // For now they only return one choice!
                    choices[0].finish_reason.to_string().to_uppercase()
                },
                Some(_) | None => {
                    "None".into()
                }
            };
        let usage: Triple = res.usage.to_triple();

        Ok(LlmReturn::new(LlmType::GPT, text, finish_reason, usage, timing, None, None))
    }
}

pub async fn get_gpt_client() -> Result<Client, Box<dyn std::error::Error + Send>> {
    // Extract API Key information
    let api_key: String =
        env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found in enviroment variables");

    // Create headers
    let mut headers: HeaderMap = HeaderMap::new();

    // Create api key header
    headers.insert(
        "Authorization",
        HeaderValue::from_str(&format!("Bearer {}", api_key))
            .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?,
    );

    get_client(headers).await
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn gpt(content: Vec<GptMessage>) {
        match call_gpt(content).await {
            Ok(ret) => { println!("{ret}"); assert!(true) },
            Err(e) => { println!("{e}"); assert!(false) },
        }
    }

    #[tokio::test]
    async fn test_call_gpt_basic() {
        let messages = vec![GptMessage::text("user", "What is the meaining of life?")];
        gpt(messages).await;
    }
    #[tokio::test]
    async fn test_call_gpt_citation() {
        let messages = 
            vec![GptMessage::text("user", "Give citations for the General theory of Relativity.")];
        gpt(messages).await;
    }
    #[tokio::test]
    async fn test_call_gpt_poem() {
        let messages = 
            vec![GptMessage::text("user", "Write a creative poem about the interplay of artificial intelligence and the human spirit and provide citations")];
        gpt(messages).await;
    }
    #[tokio::test]
    async fn test_call_gpt_logic() {
        let messages = 
            vec![GptMessage::text("user", "How many brains does an octopus have, when they have been injured and lost a leg?")];
        gpt(messages).await;
    }
    #[tokio::test]
    async fn test_call_gpt_dialogue() {
        let system = "Use a Scottish accent to answer questions";
        let mut messages = 
            vec!["How many brains does an octopus have, when they have been injured and lost a leg?".to_string()];
        let res = GptCompletion::call(&system, &messages, 0.2, false, true).await;
        println!("{res:?}");

        messages.push(res.unwrap().to_string());
        messages.push("Is a cuttle fish similar?".to_string());

        let res = GptCompletion::call(&system, &messages, 0.2, false, true).await;
        println!("{res:?}");
    }
}
