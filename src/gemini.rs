use std::collections::HashMap;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::process::Command;
use serde_derive::{Deserialize, Serialize};
use stemplate::Template;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use crate::common::*;
use crate::gpt::GptMessage;
use crate::common::{LlmType, LlmCompletion};
use crate::functions::*;

// Input structures
// Chat

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
// Container for all data to be sent
pub struct GeminiCompletion {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<FunctionDeclaration>>,
    //pub tools: Option<Vec<Function>>,
    pub safety_settings: Vec<SafetySettings>,
    pub generation_config: GenerationConfig,
}

impl GeminiCompletion {
    /// Create new Completion object
    pub fn new(contents: Vec<Content>, safety_settings: Vec<SafetySettings>, generation_config: GenerationConfig) -> Self {
        GeminiCompletion {
            contents,
            system_instruction: None,
            tools: None,
            safety_settings,
            generation_config
        }
    }

    pub fn set_system_instruction(&mut self, system: Vec<String>) {
        self.system_instruction = Some(SystemInstruction::new(system));
    }

    pub fn set_tools(&mut self, tools: Option<Vec<FunctionDeclaration>>) {
        self.tools = tools;
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct SystemInstruction {
    pub role: String,
    pub parts: Vec<Part>,
}

impl SystemInstruction {
    pub fn new_one(part: String) -> Self {
        SystemInstruction { role: "object".to_string(), parts: vec![Part::text(&part)] }
    }

    pub fn new(part: Vec<String>) -> Self {
        SystemInstruction {
            role: "object".into(),
            parts: part.iter()
                .map(|si| Part::text(si))
                .collect()
        }
    }
}

impl Default for GeminiCompletion {
    /// Create default Completion object
    fn default() -> Self {
        GeminiCompletion { 
            contents: Vec::new(),
            system_instruction: None,
            tools: None,
            safety_settings: Vec::new(),
            generation_config: GenerationConfig::new(Some(0.2), None, None, 1, Some(8192), None)
        }
    }
}

impl LlmCompletion for GeminiCompletion {
    /// Set temperature
    fn set_temperature(&mut self, temperature: f32) {
        self.generation_config.temperature = Some(temperature);
    }

    /// Add single role and single part text
    fn add_text(&mut self, role: &str, text: &str) {
        self.contents.push(Content::text(role, text));
    }

    /// Add single role with multiple strings for parts as single large content
    fn add_many_text(&mut self, role: &str, texts: &[String]) {
        self.contents.push(Content::many_text(role, texts));
    }

    /// Supply simple, 'system' content
    fn add_system(&mut self, system_prompt: &str) {
        self.contents.append(&mut Content::system(system_prompt));
    }

    /// Supply multi-parts and single 'system' content
    fn add_multi_part_system(&mut self, system_prompts: &[String]) {
        self.contents.append(&mut Content::multi_part_system(system_prompts));
    }

    /// Supply multi-context 'system' content
    fn add_systems(&mut self, system_prompts: &[String]) {
        self.contents.append(&mut Content::systems(system_prompts));
    }

    /// Supply multi-String content with user and llm alternating
    fn dialogue(&mut self, prompts: &[String], has_system: bool) {
        self.contents = Content::dialogue(prompts, has_system);
    }
    
    /// Truncate messages
    fn truncate_messages(&mut self, len: usize) {
        self.contents.truncate(len);
    }

    /// Return String of Object
    fn debug(&self) -> String where Self: std::fmt::Debug {
        format!("{:?}", self)
    }

    // Set content in precreated completion
    //fn set_content(&mut self, content: Vec<Box<dyn LlmMessage>>) {
    //    self.contents = content.iter().map(|c| *(&c as &Content)).collect();
    //}

    /// Create and call llm by supplying data and common parameters
    async fn call(system: &str, user: &[String], temperature: f32, _is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
        let model: String = std::env::var("GEMINI_MODEL").expect("GEMINI_MODEL not found in enviroment variables");

        Self::call_model(&model, system, user, temperature, _is_json, is_chat).await
    }

    /// Create and call llm by supplying data and common parameters
    async fn call_model(model: &str, system: &str, user: &[String], temperature: f32, _is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
        Self::call_model_function(model, system, user, temperature, _is_json, is_chat, None).await
    }

    /// Create and call llm with model/function by supplying data and common parameters
    async fn call_model_function(model: &str, system: &str, user: &[String], temperature: f32, _is_json: bool, is_chat: bool, function: Option<Vec<Function>>) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
        let mut contents = Vec::new();

        let system = if function.is_none() {
            system.to_string()
        } else {
            let fc = "This is a function call, find arguments and return function call";
            if !system.trim().is_empty() {
                fc.to_string()
            } else {
                format!("{fc}. {system}")
            }
        };

        if !system.is_empty() {
            contents.push(Content::text("user", &system));
            contents.push(Content::text("model", "Understood"));
        }

        user.iter()
            .enumerate()
            .for_each(|(i, c)| {
                let role = if !is_chat || i % 2 == 0 { "user" } else { "model" };

                contents.push(Content::text(role, c));
            });

//println!("{:?}", function);
        let completion = GeminiCompletion {
            contents,
            system_instruction: None,
            /*
            system_instruction: if system.is_empty() {
                None
            } else {
                Some(SystemInstruction { role: "object".to_string(), parts: vec![Part::text(&system)] })
            },
            */
            tools: Some(FunctionDeclaration::functions(function)),
            safety_settings: SafetySettings::low_block(),
            generation_config: GenerationConfig::new(Some(temperature), None, None, 1, Some(8192), None)
        };

        call_gemini_completion_model(Some(model), &completion).await
    }
}

/// This is the primary structure for loading a call. See implementation.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    pub role: String,
    //#[serde(skip_serializing_if = "Option::is_none")]
    //pub parts: Option<Vec<Part>>,
    pub parts: Vec<Part>,
}

impl Content {
    /// Supply single role and single part text
    pub fn one(role: &str, part: Part) -> Self {
        Self { role: role.into(), parts: vec![part]  }
    }

    /// Supply single role and multi-part text
    pub fn many(role: &str, parts: Vec<Part>) -> Self {
        Self { role: role.into(), parts }
    }

    /// Supply inline file data
    pub fn to_inline(role: &str, mime_type: &str, content: &[u8]) -> Vec<Self> {
        vec![Self { role: role.into(), parts: vec![Part::inline_data(mime_type, content)] }]
    }

    /// Supply file data for previously supplied file
    pub fn file(role: &str, mime_type: &str, file: &str) -> Vec<Self> {
        vec![Self { role: role.into(), parts: vec![Part::file_data(mime_type, file)] }]
    }

    pub fn message_to_content(messages: &[GptMessage]) -> Vec<Self> {
        let parts: Vec<Part> = messages.iter()
            .map(|m| Part::text(&m.content))
            .collect();

        vec![Self::many("user", parts)]
    }
}

impl LlmMessage for Content {
    /// Supply single role and single text string for Part
    fn text(role: &str, text: &str) -> Self {
        Self { role: role.into(), parts: vec![Part::text(text)] }
    }

    /// Supply single role with multi-string for iparts with single content
    fn many_text(role: &str, parts: &[String]) -> Self {
        let parts: Vec<Part> = parts.iter()
            .map(|p| Part::text(p))
            .collect();

        Self { role: role.into(), parts }
    }

    /// Supply simple, 'system' content
    fn system(system_prompt: &str) -> Vec<Self> {
        vec![Self::text("user", system_prompt), Self::text("model", "Understood")]
    }

    /// Supply multi-parts and single 'system' content
    fn multi_part_system(system_prompts: &[String]) -> Vec<Self> {
        vec![Self::many_text("user", system_prompts), Self::text("model", "Understood")]
    }

    /// Supply multi-context 'system' content
    fn systems(system_prompts: &[String]) -> Vec<Self> {
        let n = system_prompts.len() * 2;

        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    Self::text("user", &system_prompts[i / 2])
                } else {
                    Self::text("model", "Understood")
                }
            })
            .collect()
    }

    /// Supply multi-String content with user and model alternating
    fn dialogue(prompts: &[String], _has_system: bool) -> Vec<Self> {
        prompts.iter()
            .enumerate()
            .map(|(i, p)| {
                let role = if i % 2 == 0 { "user" } else { "model" };

                Self::text(role, p)
            })
            .collect()
    }

    /// Return String of Object
    fn debug(&self) -> String where Self: std::fmt::Debug {
        format!("{:?}", self)
    }
}

/// Parts to make up the content 
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
    /// Create text Part
    pub fn text(text: &str) -> Self {
        Part::Text(text.to_string())
    }

    /// Create inline data Part from data
    pub fn inline_data(mime_type: &str, data: &[u8]) -> Self {
        Part::InlineData { mime_type: mime_type.into(), data: BASE64_STANDARD.encode(data) }
    }

    /// Create inline data Part from file
    pub fn inline_file(mime_type: &str, file: &str) -> Self {
        match std::fs::read_to_string(file) {
            Ok(data) => Part::InlineData { mime_type: mime_type.into(), data: BASE64_STANDARD.encode(data.as_bytes()) },
            Err(e) => Part::InlineData { mime_type: mime_type.into(), data: BASE64_STANDARD.encode(format!("{file} not found: {e}").as_bytes()) }
        }
    }

    /// Create Part referencing previously uploaded file
    pub fn file_data(mime_type: &str, file_url: &str) -> Self {
        Part::FileData { mime_type: mime_type.into(), file_url: file_url.into() }
    }

    /// Create Offset Part for inline or uploaded files
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

/// Safety setting with helper functions
#[derive(Debug, Serialize, Clone)]
pub struct SafetySettings {
    pub category: String,
    pub threshold: String,
}

// Safety Settings, simple setting functions
impl SafetySettings {
    /// Don't Block ever i.e let everything through (Google may not like this!)
    pub fn no_block() -> Vec<Self> {
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

    /// Low threshold for blocking calls i.e let most stuff through
    pub fn low_block() -> Vec<Self> {
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

    /// Medium threshold for blocking calls i.e. block moderately 'bad' stuff
    pub fn med_block() -> Vec<Self> {
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

    /// High threshold for blocking calls i.e. block only 'bad' stuff
    pub fn high_block() -> Vec<Self> {
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

    /// Custom thresholds for 4 types of blocks
    pub fn blocks(blocks: Vec<(HarmCategory, HarmBlockThreshold)>) -> Vec<Self> {
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

/*
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
*/

// Output structures
// Chat

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    pub candidates: Vec<Candidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<Usage>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: Option<ResponseContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_ratings: Option<Vec<OutSafety>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_metadata: Option<CitationMetadata>
}

#[derive(Deserialize, Clone)]
pub struct OutSafety {
    pub category: String,
    pub probability: String,
    pub blocked: Option<bool>
}

impl std::fmt::Debug for OutSafety {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.blocked.is_some() {
            writeln!(f, "Safety Rating: {}: {} -> {:?}",
                self.category, self.probability, self.blocked)
        } else {
            Ok(())
        }
    }
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
            //writeln!(f, "Citation:")?;
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
                writeln!(f, "    Publication Date: {publication_date}")?;
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

impl std::fmt::Display for PublicationDate {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.year, self.month, self.day)?;

        Ok(())
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub prompt_token_count: usize,
    pub candidates_token_count: usize,
    pub total_token_count: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} + {} = {}", self.prompt_token_count, self.candidates_token_count, self.total_token_count)
    }
}

impl Usage {
    pub fn new() -> Self {
        Usage { prompt_token_count: 0, candidates_token_count: 0, total_token_count: 0 }
    }

    pub fn to_triple(&self) -> (usize, usize, usize) {
        (self.prompt_token_count, self.candidates_token_count, self.total_token_count)
    }
}

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
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

/// Call Large Language Model (i.e. Google Gemini) with defaults
pub async fn call_gemini(contents: Vec<Content>) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_gemini_system(None, contents).await
}

/// Call Large Language Model (i.e. Google Gemini) with 'system context' and defaults
pub async fn call_gemini_system(system: Option<&str>, contents: Vec<Content>) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_gemini_system_all(system, contents, SafetySettings::high_block(), GenerationConfig::new(Some(0.2), None, None, 1, Some(8192), None)).await
}

/// Call Large Language Model (i.e. Google Gemini) with all parameters supplied
pub async fn call_gemini_all(contents: Vec<Content>, safety: Vec<SafetySettings>, config: GenerationConfig) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_gemini_system_all(None, contents, safety, config).await
}

/// Call Large Language Model (i.e. Google Gemini) with all parameters supplied including system context
pub async fn call_gemini_system_all(system: Option<&str>, contents: Vec<Content>, safety: Vec<SafetySettings>, config: GenerationConfig) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let contents = add_system_content(system, contents);

    // Create chat completion
    let gemini_completion = GeminiCompletion::new(contents, safety, config);

    call_gemini_completion_model(None, &gemini_completion).await
}

/// Pass a pre-assembled completion object 
pub async fn call_gemini_completion(gemini_completion: &GeminiCompletion) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_gemini_completion_model(None, gemini_completion).await
}

/// Pass a pre-assembled completion object 
pub async fn call_gemini_completion_model(model: Option<&str>, gemini_completion: &GeminiCompletion) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let start = std::time::Instant::now();
    let mut env = HashMap::new();
    match model {
        None => if let Ok(gemini_model) = std::env::var("GEMINI_MODEL") {
                    env.insert("GEMINI_MODEL", gemini_model);
                },
        Some(model) => {
            env.insert("GEMINI_MODEL", model.into());
        },
    }
    let url: String = Template::new("${GEMINI_URL}").render(&env);
    let client = get_gemini_client().await?;
//println!("gemini_completion: {:?}", serde_json::to_string(&gemini_completion));

    // Extract Response
    let res = client
        .post(url)
        .json(gemini_completion)
        .send()
        .await;

    //let res: Vec<GeminiResponse> = res
    let res = res
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?
        //.json()
        .text()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?;

    let timing = start.elapsed().as_secs() as f64 + start.elapsed().subsec_millis() as f64 / 1000.0;

//println!("res: {res}");
    if res.contains("\"error\":") {
        let res: Vec<LlmError> = serde_json::from_str(&res).unwrap();

        Ok(LlmReturn::new(LlmType::GEMINI_ERROR, res[0].error.to_string(), res[0].error.to_string(), (0, 0, 0), timing, None, None))
    } else if res.contains("\"functionCall\"") {
        let found = vec![
            "candidates:content:parts:functionCall:args:${args}".to_string(),
            "candidates:content:parts:functionCall:name:${func}".to_string(),
            "usageMetadata:promptTokenCount:${in}".to_string(),
            "usageMetadata:candidatesTokenCount:${out}".to_string(),
            "usageMetadata:totalTokenCount:${total}".to_string(),
//            "usageMetadata:${usage}".to_string(),
            "candidates:finishReason:${finish}".to_string()];
        let f: serde_json::Value = serde_json::from_str(&res).unwrap();
        let h = get_functions(&f, &found);
        let funcs = unpack_functions(h.clone());
        let function_calls = serde_json::to_string(&funcs).unwrap();
//println!("{:?}", serde_json::from_str::<Vec<ParseFunction>>(&function_calls).unwrap());
        let (i, o, t) = (h.get("in").unwrap()[0].clone(), h.get("out").unwrap()[0].clone(), h.get("total").unwrap()[0].clone());
        let triple = (i.parse::<usize>().unwrap(), o.parse::<usize>().unwrap(), t.parse::<usize>().unwrap());
        let finish = h.get("finish").unwrap()[0].clone();

        Ok(LlmReturn::new(LlmType::GEMINI_TOOLS, function_calls, finish, triple, timing, None, None))
    } else {
        let res: Vec<GeminiResponse> = serde_json::from_str(&res).unwrap();

        // Now unpack it
        let text: String = res.iter()
            .map(|gr| gr.candidates.iter().map(|c| {
                if let Some(content) = &c.content {
                    if let Some(parts) = &content.parts {
                        parts.iter().map(|p| p.text.trim().to_owned() + " ").collect::<String>()
                    } else {
                        "".into()
                    }
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
        let safety_ratings: Vec<String> = res.iter()
            .map(|gr| gr.candidates.iter()
                .map(|c| if c.safety_ratings.is_some() {
                    format!("{:?}", c.safety_ratings)
                } else {
                    "".into()
                })
                .collect::<String>())
            .filter(|s| !s.is_empty() && s != "Some([, , , ])") // NOT elegant!
            .collect();
        let citations: String = res.iter()
            .map(|gr| gr.candidates.iter().map(|c| {
                if let Some(citation_metadata) = &c.citation_metadata {
                    citation_metadata.citations.iter()
                        .map(|c| c.to_string()).collect::<String>()
                } else {
                    "".into()
                }
            })
            .collect::<String>()).collect();
        let usage: Triple = res.iter()
            .fold((0, 0, 0), |mut s: Triple, g| {
                if let Some(m) = &g.usage_metadata {
                    s.0 += m.prompt_token_count;
                    s.1 += m.candidates_token_count;
                    s.2 += m.total_token_count;
                }
                s
            });

        // Remove any comments
        let text = text.lines()
            .filter(|l| !l.starts_with("```"))
            .fold(String::new(), |s, l| s + l + "\n");

        Ok(LlmReturn::new(LlmType::GEMINI, text, finish_reason, usage, timing,
                          if citations.is_empty() { None } else { Some(citations) },
                          if safety_ratings.is_empty() { None } else { Some(safety_ratings) }
                          ))
    }
}

/// Add 'system' content to other content
pub fn add_system_content(system: Option<&str>, contents: Vec<Content>) -> Vec<Content> {
    if let Some(system) = system {
        [Content::system(system), contents].concat()
    } else {
        contents
    }
}

async fn get_gemini_client() -> Result<Client, Box<dyn std::error::Error + Send>> {
    // Extract API Key information
    let output = Command::new("gcloud")
        .arg("auth")
        .arg("print-access-token")
        .output()
        .expect("Failed to execute command");

    let api_key: String = String::from_utf8_lossy(&output.stdout).trim().to_string();

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

    async fn gemini(content: Vec<Content>) {
        match call_gemini(content).await {
            Ok(ret) => { println!("{ret}"); assert!(true) },
            Err(e) => { println!("{e}"); assert!(false) },
        }
    }

    #[tokio::test]
    async fn test_call_gemini_basic() {
        let messages =
            vec![Content::text("user", "What is the meaining of life?")];
        gemini(messages).await;
    }
    #[tokio::test]
    async fn test_call_gemini_citation() {
        let messages =
            vec![Content::text("user", "Give citations for the General theory of Relativity.")];
        gemini(messages).await;
    }
    #[tokio::test]
    async fn test_call_gemini_poem() {
        let messages =
            vec![Content::text("user", "Write a creative poem about the interplay of artificial intelligence and the human spirit and provide citations")];
        gemini(messages).await;
    }
    #[tokio::test]
    async fn test_call_gemini_logic() {
        let messages =
            vec![Content::text("user", "How many brains does an octopus have, when they have been injured and lost a leg?")];
        gemini(messages).await;
    }
    #[tokio::test]
    async fn test_call_gemini_dialogue() {
        let system = "Use a Scottish accent to answer questions";
        let mut messages = 
            vec!["How many brains does an octopus have, when they have been injured and lost a leg?".to_string()];
        let res = GeminiCompletion::call(&system, &messages, 0.2, false, true).await;
        println!("{res:?}");

        messages.push(res.unwrap().to_string());
        messages.push("Is a cuttle fish similar?".to_string());

        let res = GeminiCompletion::call(&system, &messages, 0.2, false, true).await;
        println!("{res:?}");
    }
    #[tokio::test]
    async fn test_call_gemini_dialogue_model() {
        let model: String = std::env::var("GEMINI_MODEL").expect("GEMINI_MODEL not found in enviroment variables");
        let messages = vec!["Hello".to_string()];
        let res = GeminiCompletion::call_model(&model, "", &messages, 0.2, false, true).await;
        println!("{res:?}");
    }
    #[tokio::test]
    async fn test_call_function_gemini() {
        let model: String = std::env::var("GEMINI_MODEL").expect("GEMINI_MODEL not found in enviroment variables");
        let messages =  vec!["The answer is (60 * 24) * 365.25".to_string()];
        let func_def =
r#"
// Derive the value of the arithmetic expression
// expr: An arithmetic expression
fn arithmetic(expr)
"#;
        let functions = get_function_json("gemini", &[func_def]);
        let res = GeminiCompletion::call_model_function(&model, "", &messages, 0.2, false, true, functions).await;
        println!("{res:?}");

        let answer = call_actual_function(res.ok());
        println!("{answer:?}");
    }
    #[tokio::test]
    async fn test_call_function_common_gemini() {
        let messages =  vec!["The answer is (60 * 24) * 365.25".to_string()];
        let func_def =
r#"
// Recognize and derive the value of an arithmetic expression
// expr: An arithmetic expression
fn arithmetic(expr)
"#;
        // This does not work in Gemnini yet
/*
        let messages = vec!["a fruit that is red with a sweet taste".to_string()];
        let func_def2 =
r#"
// Find the color of an apple and its taste pass them to this function
// color: The color of an apple
// taste: The taste of an apple
fn apple(color, taste)
"#;
*/
        let res = call_function_llm("gemini", &messages, &[func_def]).await;
        println!("{res:?}");

        let answer = call_actual_function(res.ok());
        println!("{answer:?}");
    }
}
