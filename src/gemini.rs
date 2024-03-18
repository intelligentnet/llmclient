use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::process::Command;
use serde_derive::{Deserialize, Serialize};
use stemplate::Template;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;

// Input structures
// Chat

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiCompletion {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<FunctionDeclarations>>,
    pub safety_settings: Vec<SafetySettings>,
    pub generation_config: GenerationConfig,
}

impl GeminiCompletion {
    pub fn contents(contents: Vec<Content>, safety_settings: Vec<SafetySettings>, generation_config: GenerationConfig) -> Self {
        GeminiCompletion { contents, tools: None, safety_settings, generation_config }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

impl Content {
    pub fn many(role: &str, parts: Vec<Part>) -> Self {
        Content { role: role.into(), parts }
    }

    pub fn one(role: &str, part: Part) -> Self {
        Content { role: role.into(), parts: vec![part]  }
    }

    pub fn one_offset(role: &str, part: Part, offset: Part) -> Self {
        Content { role: role.into(), parts: vec![part, offset] }
    }

    pub fn text(role: &str, text: &str) -> Self {
        Content { role: role.into(), parts: vec![Part::text(text)] }
    }

    pub fn many_text(role: &str, parts: &[&str]) -> Self {
        let parts: Vec<Part> = parts.iter()
            .map(|p| Part::text(p))
            .collect();

        Content { role: role.into(), parts }
    }

    pub fn dialogue(prompts: &[String]) -> Vec<Self> {
        prompts.iter()
            .enumerate()
            .map(|(i, p)| {
                let role = if i % 2 == 0 { "user" } else { "model" };

                Content::text(role, p)
            })
            .collect()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub enum Part {
    Text(String),
    #[serde(rename_all = "camelCase")]
    InlineData { mime_type: String, data: String },
    #[serde(rename_all = "camelCase")]
    FileData { mime_type: String, file_url: String },
    #[serde(rename_all = "camelCase")]
    VideoMetadata { start_offset: Offset, end_offset: Offset }
}

impl Part {
    pub fn text(text: &str) -> Self {
        Part::Text(text.to_string())
    }

    pub fn inline_data(mime_type: &str, data: &[u8]) -> Self {
        Part::InlineData { mime_type: mime_type.into(), data: BASE64_STANDARD.encode(data) }
    }

    pub fn file_data(mime_type: &str, file_url: &str) -> Self {
        Part::FileData { mime_type: mime_type.into(), file_url: file_url.into() }
    }

    pub fn offset(start_secs: usize, start_nanos: usize, end_secs: usize, end_nanos: usize) -> Self {
        Part::VideoMetadata { start_offset: Offset { seconds: start_secs, nanos: start_nanos },
            end_offset: Offset { seconds: end_secs, nanos: end_nanos }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
//#[serde(rename_all = "camelCase")]
pub struct Offset {
    pub seconds: usize,
    pub nanos: usize
}

#[derive(Debug, Serialize, Clone)]
pub struct SafetySettings {
    pub category: String,
    pub threshold: String,
}

impl SafetySettings {
    pub fn no_block() -> Vec<SafetySettings> {
        vec![
            SafetySettings { category: HarmCategory::HarmCategoryHarassment.to_string(),
                threshold: HarmBlockThreshold::BlockNone.to_string() },
            SafetySettings { category: HarmCategory::HarmCategoryHateSpeech.to_string(),
                threshold: HarmBlockThreshold::BlockNone.to_string() },
            SafetySettings { category: HarmCategory::HarmCategorySexuallyExplicit.to_string(),
                threshold: HarmBlockThreshold::BlockNone.to_string() },
            SafetySettings { category: HarmCategory::HarmCategoryDangerousContent.to_string(),
                threshold: HarmBlockThreshold::BlockNone.to_string() }
        ]
    }

    pub fn low_block() -> Vec<SafetySettings> {
        vec![
            SafetySettings { category: HarmCategory::HarmCategoryHarassment.to_string(),
                threshold: HarmBlockThreshold::BlockLowAndAbove.to_string() },
            SafetySettings { category: HarmCategory::HarmCategoryHateSpeech.to_string(),
                threshold: HarmBlockThreshold::BlockLowAndAbove.to_string() },
            SafetySettings { category: HarmCategory::HarmCategorySexuallyExplicit.to_string(),
                threshold: HarmBlockThreshold::BlockLowAndAbove.to_string() },
            SafetySettings { category: HarmCategory::HarmCategoryDangerousContent.to_string(),
                threshold: HarmBlockThreshold::BlockLowAndAbove.to_string() }
        ]
    }

    pub fn med_block() -> Vec<SafetySettings> {
        vec![
            SafetySettings { category: HarmCategory::HarmCategoryHarassment.to_string(),
                threshold: HarmBlockThreshold::BlockMedAndAbove.to_string() },
            SafetySettings { category: HarmCategory::HarmCategoryHateSpeech.to_string(),
                threshold: HarmBlockThreshold::BlockMedAndAbove.to_string() },
            SafetySettings { category: HarmCategory::HarmCategorySexuallyExplicit.to_string(),
                threshold: HarmBlockThreshold::BlockMedAndAbove.to_string() },
            SafetySettings { category: HarmCategory::HarmCategoryDangerousContent.to_string(),
                threshold: HarmBlockThreshold::BlockMedAndAbove.to_string() }
        ]
    }

    pub fn high_block() -> Vec<SafetySettings> {
        vec![
            SafetySettings { category: HarmCategory::HarmCategoryHarassment.to_string(),
                threshold: HarmBlockThreshold::BlockOnlyHigh.to_string() },
            SafetySettings { category: HarmCategory::HarmCategoryHateSpeech.to_string(),
                threshold: HarmBlockThreshold::BlockOnlyHigh.to_string() },
            SafetySettings { category: HarmCategory::HarmCategorySexuallyExplicit.to_string(),
                threshold: HarmBlockThreshold::BlockOnlyHigh.to_string() },
            SafetySettings { category: HarmCategory::HarmCategoryDangerousContent.to_string(),
                threshold: HarmBlockThreshold::BlockOnlyHigh.to_string() }
        ]
    }

    pub fn blocks(blocks: Vec<(HarmCategory, HarmBlockThreshold)>) -> Vec<SafetySettings> {
        blocks.iter()
            .map(|(c, t)| SafetySettings { category: c.to_string(), threshold: t.to_string() })
            .collect()
    }
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<f32>,
    candidate_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>
}

impl GenerationConfig {
    fn new(temperature: Option<f32>, top_p: Option<f32>, top_k: Option<f32>, candidate_count: usize, max_output_tokens: Option<usize>, stop_sequences: Option<Vec<String>>) -> Self {
        GenerationConfig { temperature, top_p, top_k, candidate_count, max_output_tokens, stop_sequences }
    }
}

pub enum HarmCategory {
    HarmCategoryHarassment,
    HarmCategoryHateSpeech,
    HarmCategorySexuallyExplicit,
    HarmCategoryDangerousContent
}

impl std::fmt::Display for HarmCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            HarmCategory::HarmCategoryHarassment => write!(f, "HARM_CATEGORY_HARASSMENT"),
            HarmCategory::HarmCategoryHateSpeech => write!(f, "HARM_CATEGORY_HATE_SPEECH"),
            HarmCategory::HarmCategorySexuallyExplicit => write!(f, "HARM_CATEGORY_SEXUALLY_EXPLICIT"),
            HarmCategory::HarmCategoryDangerousContent => write!(f, "HARM_CATEGORY_DANGEROUS_CONTENT")
        }
    }
}

pub enum HarmBlockThreshold {
    BlockNone,
    BlockLowAndAbove,
    BlockMedAndAbove,
    BlockOnlyHigh,
}

impl std::fmt::Display for HarmBlockThreshold {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            HarmBlockThreshold::BlockNone => write!(f, "BLOCK_NONE"),
            HarmBlockThreshold::BlockLowAndAbove => write!(f, "BLOCK_LOW_AND_ABOVE"),
            HarmBlockThreshold::BlockMedAndAbove => write!(f, "BLOCK_MED_AND_ABOVE"),
            HarmBlockThreshold::BlockOnlyHigh => write!(f, "BLOCK_ONLY_HIGH")
        }
    }
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclarations {
    name: String,
    description: String,
    parameters: Option<bool>
}

impl FunctionDeclarations {
    pub fn new(name: &str, description: &str, parameters: Option<bool>) -> Self {
        FunctionDeclarations { name: name.into(), description: description.into(), parameters }
    }
}

// Output structures
// Chat

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    pub candidates: Vec<Candidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<Metadata>, // TODO: Fix as not parsing!
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: ResponseContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub safety_ratings: Vec<OutSafety>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_metadata: Option<CitationMetadata>
}

#[derive(Debug, Deserialize, Clone)]
pub struct OutSafety {
    pub category: String,
    pub probability: String,
    pub blocked: Option<bool>
}

#[derive(Debug, Deserialize, Clone)]
pub struct CitationMetadata {
    pub citations: Vec<Citation>
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Citation {
    pub start_index: Option<usize>,
    pub end_index: Option<usize>,
    pub uri: Option<String>,
    pub license: Option<String>,
    pub publication_date: Option<PublicationDate>
}

impl std::fmt::Display for Citation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(uri) = &self.uri {
            writeln!(f, "Citation:")?;
            match BASE64_STANDARD.decode(uri) {
                Ok(uri) => writeln!(f, "    Uri: {:?}", String::from_utf8(uri)),
                Err(_) => Ok(writeln!(f, "    Uri: {}", uri)?)
            }?;

            if let Some(start_index) = self.start_index {
                if let Some(end_index) = self.end_index {
                    writeln!(f, "    Index range: {start_index} - {end_index}")?;
                }
            }
            if let Some(license) = &self.license {
                writeln!(f, "    License: {license}")?;
            }
            if let Some(publication_date) = &self.publication_date {
                writeln!(f, "    Publication Date: {publication_date:?}")?;
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct PublicationDate {
    pub year: usize,
    pub month: usize,
    pub day: usize,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Metadata {
    pub prompt_token_count: usize,
    pub candidates_token_count: usize,
    pub total_token_count: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseContent {
    pub role: String,
    pub parts: Option<Vec<ResponsePart>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponsePart {
    pub text: String,
}

// Call Large Language Model (i.e. Google Gemini)
pub async fn call_gemini(contents: Vec<Content>) -> Result<(String, String, String), Box<dyn std::error::Error + Send>> {
    let url: String = Template::new("${GEMINI_URL}").render_env();
    let client = get_client().await?;

    // Create chat completion
    let gemini_completion: GeminiCompletion = GeminiCompletion {
        contents, 
        tools: None,
        safety_settings: SafetySettings::high_block(),
        generation_config: GenerationConfig::new(Some(0.2), None, None, 1, Some(8192), None)
    };

    // Extract Response
    let res = client
        .post(url)
        .json(&gemini_completion)
        .send()
        .await;
    let res: Vec<GeminiResponse> = res
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?
        .json()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?;

    // Now unpack it
    let text: String = res.iter()
        .map(|gr| gr.candidates.iter().map(|c| {
            if let Some(parts) = &c.content.parts {
                parts.iter().map(|p| p.text.trim().to_owned() + " ").collect::<String>()
            } else {
                "".into()
            }
        })
        .collect::<String>()).collect();
    let finish_reason: String = res.iter()
        .map(|gr| gr.candidates.iter().map(|c| {
            if let Some(finish) = &c.finish_reason { finish.clone() } else { "".into() }
        })
        .collect::<String>()).collect();
    let citations: String = res.iter()
        .map(|gr| gr.candidates.iter().map(|c| {
            if let Some(citation_metadata) = &c.citation_metadata {
                citation_metadata.citations.iter().map(|c| c.to_string()).collect::<String>()
            } else {
                "".into()
            }
        })
        .collect::<String>()).collect();

    // Remove any comments
    let text = text.lines()
        .filter(|l| !l.starts_with("```"))
        .fold(String::new(), |s, l| s + l + "\n");

    Ok((text, finish_reason, citations))
}

async fn get_client() -> Result<Client, Box<dyn std::error::Error + Send>> {
    // Extract API Key information
    let output = Command::new("gcloud")
        .arg("auth")
        .arg("print-access-token")
        .output()
        .expect("Failed to execute command");

    let api_key: String = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // Create headers
    let mut headers: HeaderMap = HeaderMap::new();

    // We would like json
    headers.insert(
        "Content-Type",
        HeaderValue::from_str("appication/json; charset=utf-8")
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

pub fn texts(role: &str, texts: &[&str]) -> Vec<Content> {
    let parts: Vec<Part> = texts.iter().map(|t| Part::text(t)).collect();

    vec![Content { role: role.into(), parts }]
}

pub fn text_to_inline(role: &str, mime_type: &str, content: &[u8]) -> Vec<Content> {
    vec![Content { role: role.into(), parts: vec![Part::inline_data(mime_type, content)] }]
}

pub fn text_to_file(role: &str, mime_type: &str, file: &str) -> Vec<Content> {
    vec![Content { role: role.into(), parts: vec![Part::file_data(mime_type, file)] }]
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn gemini(content: Vec<Content>) {
        match call_gemini(content).await {
            Ok((text, finish_reason, citations)) => {
                println!("{}", text);
                if !finish_reason.is_empty() {
                    println!("Finish Reason: {}", finish_reason);
                }
                if !citations.is_empty() {
                    println!("{}", citations);
                }
                assert!(true);
            },
            Err(e) => { println!("{e}"); assert!(false) },
        }
    }

    #[tokio::test]
    async fn test_call_gemini_basic() {
        let messages: Vec<Content> = 
            vec![Content::text("user", "What is the meaining of life?")];
        gemini(messages).await;
    }
    #[tokio::test]
    async fn test_call_gemini_citation() {
        let messages: Vec<Content> = 
            vec![Content::text("user", "Give citations for the General theory of Relativity.")];
        gemini(messages).await;
    }
    #[tokio::test]
    async fn test_call_gemini_poem() {
        let messages: Vec<Content> = 
            vec![Content::text("user", "Write a creative poem about the interplay of artificial intelligence and the human spirit and provide citations")];
        gemini(messages).await;
    }
    #[tokio::test]
    async fn test_call_gemini_logic() {
        let messages: Vec<Content> = 
            vec![Content::text("user", "How many brains does an octopus have, when they have been injured and lost a leg?")];
        gemini(messages).await;
    }
}
