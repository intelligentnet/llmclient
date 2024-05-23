use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use serde_derive::{Deserialize, Serialize};
use crate::common::*;
use crate::gpt::GptMessage as MistralMessage;

// Input structures
// Chat
#[derive(Debug, Serialize, Clone)]
pub struct MistralCompletion {
    pub model: String,
    pub messages: Vec<MistralMessage>,
    pub temperature: f32,
    //pub top_p: f32,
    pub max_tokens: usize,
    //pub stream: bool,
    //pub random_seed: i32,
}

impl MistralCompletion {
    /// Create chat completion
    pub fn new(messages: Vec<MistralMessage>, temperature: f32, max_tokens: usize, _is_json: bool) -> Self {
        let model: String = env::var("MISTRAL_MODEL").expect("MISTRAL_MODEL not found in enviroment variables");

        MistralCompletion {
            model,
            messages,
            temperature,
            max_tokens,
        }
    }

    pub fn set_model(&mut self, model: &str) {
        self.model = model.into();
    }

    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens;
    }

    /// Add a single new message
    pub fn add_message(&mut self, message: &MistralMessage) {
        self.messages.push(message.clone());
    }

    /// Add many new messages
    pub fn add_messages(&mut self, messages: &[MistralMessage]) {
        messages.iter().for_each(|m| self.messages.push(m.clone()));
    }
}

impl Default for MistralCompletion {
    /// Create default chat completion
    fn default() -> Self {
        let model: String = env::var("MISTRAL_MODEL").expect("MISTRAL_MODEL not found in enviroment variables");

        MistralCompletion {
            model,
            messages: Vec::new(),
            temperature: 0.2,
            max_tokens: 4096
        }
    }
}

impl LlmCompletion for MistralCompletion {
    /// Set temperature
    fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    /// Add single role and single part text
    fn add_text(&mut self, role: &str, text: &str) {
        self.messages.push(MistralMessage::text(role, text));
    }

    /// Add single role with multiple strings for parts as single large content
    fn add_many_text(&mut self, role: &str, texts: &[String]) {
        self.messages.push(MistralMessage::many_text(role, texts));
    }

    /// Supply simple, 'system' content
    fn add_system(&mut self, system_prompt: &str) {
        self.messages.append(&mut MistralMessage::system(system_prompt));
    }

    /// Supply multi-parts and single 'system' content
    fn add_multi_part_system(&mut self, system_prompts: &[String]) {
        self.messages.append(&mut MistralMessage::multi_part_system(system_prompts));
    }

    /// Supply multi-context 'system' content
    fn add_systems(&mut self, system_prompts: &[String]) {
        self.messages.append(&mut MistralMessage::systems(system_prompts));
    }

    /// Supply multi-String content with user and llm alternating
    fn dialogue(&mut self, prompts: &[String], has_system: bool) {
        self.messages = MistralMessage::dialogue(prompts, has_system);
    }
    
    /// Truncate messages
    fn truncate_messages(&mut self, len: usize) {
        self.messages.truncate(len);
    }

    /// Return String of Object
    fn debug(&self) -> String where Self: std::fmt::Debug {
        format!("{:?}", self)
    }

    /// Create and call llm by supplying data and common parameters
    async fn call(system: &str, user: &[String], temperature: f32, _is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
        let model: String = env::var("MISTRAL_MODEL").expect("MISTRAL_MODEL not found in enviroment variables");

        Self::call_model(&model, system, user, temperature, _is_json, is_chat).await
    }

    /// Create and call llm by supplying data and common parameters
    async fn call_model(model: &str, system: &str, user: &[String], temperature: f32, _is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
        let mut messages = Vec::new();

        if !system.is_empty() {
            messages.push(MistralMessage { role: "system".into(), content: system.into() });
        }

        user.iter()
            .enumerate()
            .for_each(|(i, c)| {
                let role = if !is_chat || i % 2 == 0 { "user" } else { "assistant" };

                messages.push(MistralMessage { role: role.into(), content: c.to_string() });
            });

        let completion = MistralCompletion {
            model: model.into(),
            messages,
            temperature,
            max_tokens: 4096
        };

        call_mistral_completion(&completion).await
    }
}

// Output structures
// Chat
#[derive(Debug, Deserialize)]
pub struct MistralResponse {
    pub id: String,
    //pub object: String,
    pub created: usize,
    pub model: String,
    pub choices: Option<Vec<MistralChoice>>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct MistralChoice {
    //pub index: usize,
    pub message: MistralMessage,
    pub finish_reason: String,
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

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} + {} = {}", self.prompt_tokens, self.completion_tokens, self.total_tokens)
    }
}

/// Call Mistral with some messages
pub async fn call_mistral(messages: Vec<MistralMessage>) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_mistral_all(messages, 0.2, 4096).await
}

/// Call Mistral with some messages and temperature
pub async fn call_mistral_temperature(messages: Vec<MistralMessage>, temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_mistral_all(messages, temperature, 4096).await
}

/// Call Mistral with some messages and max_tokens
pub async fn call_mistral_max_tokens(messages: Vec<MistralMessage>, max_tokens: usize) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_mistral_all(messages, 0.2, max_tokens).await
}

/// Call Mistral with some messages, temperature and max_tokens
pub async fn call_mistral_all(messages: Vec<MistralMessage>, temperature: f32, max_tokens: usize) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let mistral_completion = MistralCompletion::new(messages, temperature, max_tokens, false);

    call_mistral_completion(&mistral_completion).await
}

pub async fn call_mistral_completion(mistral_completion: &MistralCompletion) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let start = std::time::Instant::now();
    // Endpoint
    let url: String =
        env::var("MISTRAL_URL").expect("MISTRAL_URL not found in enviroment variables");

    let client = get_mistral_client().await?;

    // Extract API Response
    let res = client
        .post(url)
        .json(&mistral_completion)
        .send()
        .await;
    //let res: MistralRespinse = res
    let res = res
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?
        //.json()
        .text()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?;

    let timing = start.elapsed().as_secs() as f64 + start.elapsed().subsec_millis() as f64 / 1000.0;

    if res.contains("\"error\":") {
        let res: LlmError = serde_json::from_str(&res).unwrap();

        Ok(LlmReturn::new(LlmType::MISTRAL_ERROR, res.error.to_string(), res.error.to_string(), (0, 0, 0), timing, None, None))
    } else {
        let res: MistralResponse = serde_json::from_str::<MistralResponse>(&res).unwrap();

        // Send Response
        let (text, finish_reason) =
            match res.choices {
                Some(choices) => {
                    if choices.len() > 1 {
                        eprintln!("There are {:?} choices available now. Code needs to change to reflect this.", choices.len());
                    }
                    let text = choices[0].message.content.clone();
                    let finish_reason = choices[0].finish_reason.to_uppercase().clone();
                    let text = text.lines().filter(|l| !l.starts_with("```")).fold(String::new(), |s, l| s + l + "\n");

                    (text, finish_reason)
                },
                None => {
                    ("None".into(), "ERROR".into())
                }
            };

        let usage: Triple = res.usage.to_triple();
        let timing = start.elapsed().as_secs() as f64 + start.elapsed().subsec_millis() as f64 / 1000.0;

        Ok(LlmReturn::new(LlmType::MISTRAL, text, finish_reason, usage, timing, None, None))
    }
}

async fn get_mistral_client() -> Result<Client, Box<dyn std::error::Error + Send>> {
    // Extract API Key information
    let api_key: String =
        env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not found in enviroment variables");

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

    async fn mistral(content: Vec<MistralMessage>) {
        match call_mistral(content).await {
            Ok(ret) => { println!("{ret}"); assert!(true) },
            Err(e) => { println!("{e}"); assert!(false) },
        }
    }

    #[tokio::test]
    async fn test_call_mistral_basic() {
        let messages: Vec<MistralMessage> = vec![MistralMessage { role: "user".into(), content: "What is the meaining of life?".into() }];

        mistral(messages).await;
    }
    #[tokio::test]
    async fn test_call_mistral_citation() {
        let messages = 
            vec![MistralMessage::text("user", "Give citations for the General theory of Relativity.")];
        mistral(messages).await;
    }
    #[tokio::test]
    //#[serial]
    async fn test_call_mistral_poem() {
        let messages = 
            vec![MistralMessage::text("user", "Write a creative poem about the interplay of artificial intelligence and the human spirit and provide citations")];
        mistral(messages).await;
    }
    #[tokio::test]
    //#[serial]
    async fn test_call_mistral_logic() {
        let messages = 
            vec![MistralMessage::text("user", "How many brains does an octopus have, when they have been injured and lost a leg?")];
        mistral(messages).await;
    }
    #[tokio::test]
    async fn test_call_mistral_dialogue() {
        let system = "Use a Scottish accent to answer questions";
        let mut messages = 
            vec!["How many brains does an octopus have, when they have been injured and lost a leg?".to_string()];
        let res = MistralCompletion::call(&system, &messages, 0.2, false, true).await;
        println!("{res:?}");

        messages.push(res.unwrap().to_string());
        messages.push("Is a cuttle fish similar?".to_string());

        let res = MistralCompletion::call(&system, &messages, 0.2, false, true).await;
        println!("{res:?}");
    }
    #[tokio::test]
    async fn test_call_mistral_dialogue_model() {
        let model: String = std::env::var("MISTRAL_MODEL").expect("MISTRAL_MODEL not found in enviroment variables");
        let messages = vec!["Hello".to_string()];
        let res = MistralCompletion::call_model(&model, "", &messages, 0.2, false, true).await;
        println!("{res:?}");
    }
}
