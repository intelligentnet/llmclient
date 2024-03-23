use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use serde_derive::{Deserialize, Serialize};
use crate::common::{LlmType, LlmReturn};

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
    fn new(model: String, messages: Vec<GptMessage>, temperature: f32, is_json: bool) -> Self {
        GptCompletion {
            model,
            messages,
            temperature,
            response_format: ResponseFormat::new(is_json)
        }
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

impl GptMessage {
    /// Supply single role and single part text
    pub fn text(role: &str, content: &str) -> Self {
        GptMessage { role: role.into(), content: content.into() }
    }

    /// Supply single role with multi-string for iparts with single content
    pub fn many_text(role: &str, prompt: &[String]) -> Self {
        let prompt: String = 
            prompt.iter()
                .fold(String::new(), |mut s, p| {
                    s.push_str(if s.is_empty() { "" } else { "\n" });
                    s.push_str(p);

                    s
                });

        GptMessage { role: role.into(), content: prompt }
    }

    pub fn system(system_prompt: &str) -> Vec<Self> {
        vec![GptMessage::text("system", system_prompt)]
    }

    pub fn multi_part_system(system_prompts: &[String]) -> Vec<Self> {
        vec![GptMessage::many_text("system", system_prompts)]
    }

    /// Supply multi-context 'system' content
    pub fn systems(system_prompts: &[String]) -> Vec<Self> {
        system_prompts.iter()
            .map(|sp| GptMessage::text("system", &sp))
            .collect()
    }

    /// Supply multi-String content with user and model alternating
    pub fn dialogue(prompts: &[String]) -> Vec<Self> {
        prompts.iter()
            .enumerate()
            .map(|(i, p)| {
                let role = if i % 2 == 0 { "user" } else { "assistant" };

                GptMessage::text(role, p)
            })
            .collect()
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
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tokens: {} + {} = {}", self.prompt_tokens, self.completion_tokens, self.total_tokens)
    }
}

// Call Large Language Model (i.e. GPT-4)
pub async fn call_gpt(messages: Vec<GptMessage>) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_gpt_all(messages, 0.2, false).await
}

pub async fn call_gpt_json(messages: Vec<GptMessage>, is_json: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_gpt_all(messages, 0.2, is_json).await
}

pub async fn call_gpt_temperature(messages: Vec<GptMessage>, temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_gpt_all(messages, temperature, false).await
}

pub async fn call_gpt_all(messages: Vec<GptMessage>, temperature: f32, is_json: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let start = std::time::Instant::now();
    let model: String = env::var("GPT_VERSION").expect("GPT_VERSION not found in enviornment variables");
    // Confirm endpoint
    let url: String = env::var("GPT_CHAT_URL").expect("GPT_CHAT_URL not found in enviornment variables");

    let client = get_client().await?;

    // Create chat completion
    let chat_completion = GptCompletion::new(model, messages, temperature, is_json);

    // Extract API Response
    let res = client
        .post(url)
        .json(&chat_completion)
        .send()
        .await;
    let res: GptResponse = res
    //let res = res
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?
        .json()
        //.text()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?;
//println!("{:?}", res);
//let res: GptResponse = serde_json::from_str::<GptResponse>(&res).unwrap();

    // Send Response
    let text: String =
        match res.choices {
            Some(ref choices) if choices.len() >= 1 => {
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
            Some(ref choices) if choices.len() >= 1 => {
                // For now they only return one choice!
                choices[0].finish_reason.to_string().to_uppercase()
            },
            Some(_) | None => {
                "None".into()
            }
        };
    let usage: String = res.usage.to_string();
    let timing = format!("{:?}", start.elapsed());

    Ok(LlmReturn::new(LlmType::GPT, text, finish_reason, usage, timing, None, None))
}

pub async fn get_client() -> Result<Client, Box<dyn std::error::Error + Send>> {
    // Extract API Key information
    let api_key: String =
        env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found in enviornment variables");

    // Create headers
    let mut headers: HeaderMap = HeaderMap::new();

    // We would like json
    headers.insert(
        "Content-Type",
        HeaderValue::from_str("appication/json")
            .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?,
    );
    // Create api key header
    headers.insert(
        "Authorization",
        HeaderValue::from_str(&format!("Bearer {}", api_key))
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
}
