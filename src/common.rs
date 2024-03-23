#[derive(Debug)]
pub enum LlmType  {
    GEMINI,
    GPT
}

impl std::fmt::Display for LlmType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            LlmType::GEMINI => write!(f, "GEMINI"),
            LlmType::GPT => write!(f, "GPT"),
        }
    }
}

#[derive(Debug)]
pub struct LlmReturn {
    pub llm_type: LlmType,
    pub text: String,
    pub finish_reason: String,
    pub usage: String,
    pub timing: String,
    pub citations: Option<String>,
    pub safety_ratings: Option<Vec<String>>,
}

impl LlmReturn {
    pub fn new(llm_type: LlmType, text: String, finish_reason: String, usage: String, timing: String, citations: Option<String>, safety_ratings: Option<Vec<String>>) -> Self {
        LlmReturn { llm_type, text, finish_reason, usage, timing, citations, safety_ratings }
    }
}

impl std::fmt::Display for LlmReturn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        println!("---------- {} ----------", self.llm_type);
        let _ = writeln!(f, "{}", self.text);
        if !self.finish_reason.is_empty() && self.finish_reason != "STOP" {
            let _ = println!("Finish Reason: {}", self.finish_reason);
        }
        println!("Usage: {}", self.usage);
        println!("Timing: {}", self.timing);
        if let Some(ref citations) = self.citations {
            println!("Citations:\n{}", citations);
        }
        if let Some(ref safety_ratings) = self.safety_ratings {
            println!("Safety Settings: {:?}", safety_ratings);
        }

        Ok(())
    }
}

