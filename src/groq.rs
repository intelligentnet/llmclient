use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::env;
use serde_derive::{Deserialize, Serialize};
use crate::common::*;
use crate::gpt::GptMessage as GroqMessage;
use crate::functions::*;

// Input structures
// Chat

/// Main chat object
/// Note: Same interface to OpenAI so duplication of code.
/// This will probably change so tolerated for now.
#[derive(Debug, Serialize, Clone)]
pub struct GroqCompletion {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<FunctionCall>>,
    pub messages: Vec<GroqMessage>,
    pub response_format: ResponseFormat,
    pub temperature: f32,
}

impl GroqCompletion {
    /// Create chat completion
    pub fn new(messages: Vec<GroqMessage>, temperature: f32, is_json: bool) -> Self {
        let model: String = env::var("GROQ_MODEL").expect("GROQ_MODEL not found in enviroment variables");

        GroqCompletion {
            model,
            tools: None,
            messages,
            temperature,
            response_format: ResponseFormat::new(is_json)
        }
    }

    pub fn set_model(&mut self, model: &str) {
        self.model = model.into();
    }

    pub fn set_tools(&mut self, tools: Option<Vec<FunctionCall>>) {
        self.tools = tools;
    }

    pub fn set_response_format(&mut self, response_format: &ResponseFormat) {
        self.response_format = response_format.clone();
    }

    /// Add a single new message
    pub fn add_message(&mut self, message: &GroqMessage) {
        self.messages.push(message.clone());
    }

    /// Add many new messages
    pub fn add_messages(&mut self, messages: &[GroqMessage]) {
        messages.iter().for_each(|m| self.messages.push(m.clone()));
    }
}

impl Default for GroqCompletion {
    /// Create default chat completion
    fn default() -> Self {
        let model: String = env::var("GROQ_MODEL").expect("GROQ_MODEL not found in enviroment variables");

        GroqCompletion {
            model,
            tools: None,
            messages: Vec::new(),
            temperature: 0.2,
            response_format: ResponseFormat::new(false)
        }
    }
}

impl LlmCompletion for GroqCompletion {
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
        self.messages.push(GroqMessage::text(role, text));
    }

    /// Add single role with multiple strings for parts as single large content
    fn add_many_text(&mut self, role: &str, texts: &[String]) {
        self.messages.push(GroqMessage::many_text(role, texts));
    }

    /// Supply simple, 'system' content
    fn add_system(&mut self, system_prompt: &str) {
        self.messages.append(&mut GroqMessage::system(system_prompt));
    }

    /// Supply multi-parts and single 'system' content
    fn add_multi_part_system(&mut self, system_prompts: &[String]) {
        self.messages.append(&mut GroqMessage::multi_part_system(system_prompts));
    }

    /// Supply multi-context 'system' content
    fn add_systems(&mut self, system_prompts: &[String]) {
        self.messages.append(&mut GroqMessage::systems(system_prompts));
    }

    /// Supply multi-String content with user and llm alternating
    fn dialogue(&mut self, prompts: &[String], has_system: bool) {
        self.messages = GroqMessage::dialogue(prompts, has_system);
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

    /// Create and call llm by supplying data and common parameters
    async fn call(system: &str, user: &[String], temperature: f32, is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
        let model: String = env::var("GROQ_MODEL").expect("GROQ_MODEL not found in enviroment variables");

        Self::call_model(&model, system, user, temperature, is_json, is_chat).await
    }

    /// Create and call llm with model by supplying data and common parameters
    async fn call_model(model: &str, system: &str, user: &[String], temperature: f32, is_json: bool, is_chat: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
        Self::call_model_function(model, system, user, temperature, is_json, is_chat, None).await
    }

    /// Create and call llm with model/function by supplying data and common parameters
    async fn call_model_function(model: &str, system: &str, user: &[String], temperature: f32, is_json: bool, is_chat: bool, function: Option<Vec<Function>>) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
        let mut messages = Vec::new();

        if !system.is_empty() {
            messages.push(GroqMessage { role: "system".into(), content: system.into() });
        }

        user.iter()
            .enumerate()
            .for_each(|(i, c)| {
                let role = if !is_chat || i % 2 == 0 { "user" } else { "assistant" };

                messages.push(GroqMessage { role: role.into(), content: c.to_string() });
            });

        let completion = GroqCompletion {
            model: model.into(),
            tools: Some(FunctionCall::functions(function)),
            messages,
            temperature,
            response_format: ResponseFormat::new(is_json)
        };

        call_groq_completion(&completion).await
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

// Output structures
// Chat
#[derive(Debug, Deserialize)]
pub struct GroqResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub usage: Usage,
    pub choices: Option<Vec<GroqChoice>>,
}

#[derive(Debug, Deserialize)]
pub struct GroqChoice {
    pub message: GroqMessage,
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

/// Call GROQ with some messages
pub async fn call_groq(messages: Vec<GroqMessage>) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_groq_all(messages, 0.2, false).await
}

/// Call GROQ with some messages and option for Json
pub async fn call_groq_json(messages: Vec<GroqMessage>, is_json: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_groq_all(messages, 0.2, is_json).await
}

/// Call GROQ with some messages and temperature
pub async fn call_groq_temperature(messages: Vec<GroqMessage>, temperature: f32) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    call_groq_all(messages, temperature, false).await
}

/// Call GROQ with some messages, option for Json and temperature
pub async fn call_groq_all(messages: Vec<GroqMessage>, temperature: f32, is_json: bool) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    // Create chat completion
    let groq_completion = GroqCompletion::new(messages, temperature, is_json);

    call_groq_completion(&groq_completion).await
}

/// Call Claude with pre-assembled completion
pub async fn call_groq_completion(groq_completion: &GroqCompletion) -> Result<LlmReturn, Box<dyn std::error::Error + Send>> {
    let start = std::time::Instant::now();
    // Confirm endpoint
    let url: String = env::var("GROQ_CHAT_URL").expect("GROQ_CHAT_URL not found in enviroment variables");

    let client = get_groq_client().await?;

//println!("{:?}", serde_json::to_string(&groq_completion));
    // Extract API Response
    let res = client
        .post(url)
        .json(&groq_completion)
        .send()
        .await;
    //let res: GroqResponse = res
    let res = res
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?
        //.json()
        .text()
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send> { Box::new(e) })?;

    let timing = start.elapsed().as_secs() as f64 + start.elapsed().subsec_millis() as f64 / 1000.0;

//println!("{res}");
    if res.contains("\"error:\"") {
        let ret: Result<LlmError,_> = serde_json::from_str(&res);

        match ret {
            Ok(res) => 
                Ok(LlmReturn::new(LlmType::GROQ_ERROR, res.error.to_string(), res.error.to_string(), (0, 0, 0), timing, None, None)),
            Err(e) => {
                eprintln!("Error: {:?}", res);

                Ok(LlmReturn::new(LlmType::GROQ_ERROR, e.to_string(), e.to_string(), (0, 0, 0), timing, None, None))
            }
        }
    } else if res.contains("\"error\"") {
        Ok(LlmReturn::new(LlmType::GROQ_ERROR, res.to_string(), res.to_string(), (0, 0, 0), timing, None, None))
    } else if res.contains("\"arguments\":") {
        let found = vec!["choices:message:tool_calls:function:arguments:${args}".to_string(),
            "choices:message:tool_calls:function:name:${func}".to_string(),
            "usage:prompt_tokens:${in}".to_string(),
            "usage:completion_tokens:${out}".to_string(),
            "usage:total_tokens:${total}".to_string(),
//            "usage:${usage}".to_string(),
            "choices:finish_reason:${finish}".to_string()];
        let f: serde_json::Value = serde_json::from_str(&res).unwrap();
        let h = get_functions(&f, &found);
        let funcs = unpack_functions(h.clone());
        let function_calls = serde_json::to_string(&funcs).unwrap();
        let (i, o, t) = (h.get("in").unwrap()[0].clone(), h.get("out").unwrap()[0].clone(), h.get("total").unwrap()[0].clone());
        let triple = (i.parse::<usize>().unwrap(), o.parse::<usize>().unwrap(), t.parse::<usize>().unwrap());
        let finish = h.get("finish").unwrap()[0].clone();

        Ok(LlmReturn::new(LlmType::GROQ_TOOLS, function_calls, finish, triple, timing, None, None))
    } else {
        let res: GroqResponse = serde_json::from_str::<GroqResponse>(&res).unwrap();

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

        Ok(LlmReturn::new(LlmType::GROQ, text, finish_reason, usage, timing, None, None))
    }
}

async fn get_groq_client() -> Result<Client, Box<dyn std::error::Error + Send>> {
    // Extract API Key information
    let api_key: String =
        env::var("GROQ_API_KEY").expect("GROQ_API_KEY not found in enviroment variables");

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

    async fn groq(content: Vec<GroqMessage>) {
        match call_groq(content).await {
            Ok(ret) => { println!("{ret}"); assert!(true) },
            Err(e) => { println!("{e}"); assert!(false) },
        }
    }

    #[tokio::test]
    async fn test_call_groq_basic() {
        let messages = vec![GroqMessage::text("user", "What is the meaining of life?")];
        groq(messages).await;
    }
    #[tokio::test]
    async fn test_call_groq_citation() {
        let messages = 
            vec![GroqMessage::text("user", "Give citations for the General theory of Relativity.")];
        groq(messages).await;
    }
    #[tokio::test]
    async fn test_call_groq_poem() {
        let messages = 
            vec![GroqMessage::text("user", "Write a creative poem about the interplay of artificial intelligence and the human spirit and provide citations")];
        groq(messages).await;
    }
    #[tokio::test]
    async fn test_call_groq_logic() {
        let messages = 
            vec![GroqMessage::text("user", "How many brains does an octopus have, when they have been injured and lost a leg?")];
        groq(messages).await;
    }
    #[tokio::test]
    async fn test_call_groq_dialogue() {
        let system = "Use a Scottish accent to answer questions";
        let mut messages = 
            vec!["How many brains does an octopus have, when they have been injured and lost a leg?".to_string()];
        let res = GroqCompletion::call(&system, &messages, 0.2, false, true).await;
        println!("{res:?}");

        messages.push(res.unwrap().to_string());
        messages.push("Is a cuttle fish similar?".to_string());

        let res = GroqCompletion::call(&system, &messages, 0.2, false, true).await;
        println!("{res:?}");
    }
    #[tokio::test]
    async fn test_call_groq_dialogue_model() {
        let model: String = std::env::var("GROQ_MODEL").expect("GROQ_MODEL not found in enviroment variables");
        let messages = vec!["Hello".to_string()];
        let res = GroqCompletion::call_model(&model, "", &messages, 0.2, false, true).await;
        println!("{res:?}");
    }
    #[tokio::test]
    async fn test_call_function_groq() {
        let model: String = std::env::var("GROQ_MODEL").expect("GROQ_MODEL not found in enviroment variables");
        let messages =  vec!["The answer is (60 * 24) * 365.25".to_string()];
        let func_def =
r#"
// Derive the value of the arithmetic expression
// expr: An arithmetic expression
fn arithmetic(expr)
"#;
        let functions = get_function_json("groq", &[func_def]);
        let res = GroqCompletion::call_model_function(&model, "", &messages, 0.2, false, true, functions).await;
        println!("{res:?}");

        let answer = call_actual_function(res.ok());
        println!("{answer:?}");
    }
    #[tokio::test]
    async fn test_call_function_common_groq() {
        //let messages =  vec!["The answer is (60 * 24) * 365.25".to_string()];
        let messages = vec!["a fruit that is blue with a sour tast".to_string()];
        let func_def =
r#"
// Derive the value of the arithmetic expression
// expr: An arithmetic expression
fn arithmetic(expr)
"#;
        let func_def2 =
r#"
// Find the color of an apple and it's taste pass them to this function.
// color: The color of an apple
// taste: The taste of an apple
fn apple(color, taste)
"#;
        let res = call_function_llm("groq", &messages, &[func_def, func_def2]).await;
        println!("{res:?}");

        let answer = call_actual_function(res.ok());
        println!("{answer:?}");
    }
}
