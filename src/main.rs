use crossterm::{
    style::{Color, ResetColor, SetForegroundColor},
    ExecutableCommand,
};
use std::io::{stdin, stdout};
use llmclient::gemini::{call_gemini, Content};

#[tokio::main]
async fn main() {
    let mut prompts: Vec<String> = Vec::new();

    loop {
        let prompt = get_user_response("Your question (Multiple lines and then ^D [or ^Z on Windows]):");

        if prompt.is_empty() || prompt.to_lowercase() == "quit" || prompt.to_lowercase() == "exit" {
            break;
        } else if prompt.to_lowercase() == "new" {
            prompts = Vec::new();

            continue;
        } else if prompt.to_lowercase() == "show" {
            println!("{:?}", prompts);

            continue;
        }

        prompts.push(prompt);

        let res: Result<String, Box<dyn std::error::Error + Send>> = llm_chat(&prompts[..]).await;

        match res {
            Ok(res_str) => {

                prompts.push(res_str.clone());

                println!("> {}", res_str);
            },
            Err(e) => {
                println!("Error (aborting): {}", e);

                break;
            }
        }
    }
}

async fn gemini(content: Vec<Content>) -> String{
    match call_gemini(content).await {
        Ok((text, finish_reason, citations)) => {
            if !finish_reason.is_empty() && finish_reason != "STOP" {
                println!("Finish Reason: {}", finish_reason);
            }
            if !citations.is_empty() {
                println!("{}", citations);
            }

            text
        },
        Err(e) => { e.to_string() },
    }
}

async fn llm_chat(prompts: &[String]) -> Result<String, Box<dyn std::error::Error + Send>> {
    let content = Content::dialogue(prompts);

    let ret = gemini(content).await;

    Ok(ret)
}

// Get user request
fn get_user_response(question: &str) -> String {
    let mut stdout: std::io::Stdout = stdout();

    // Print the question in a specific color
    stdout.execute(SetForegroundColor(Color::Blue)).unwrap();
    println!();
    println!("{}", question);

    // Reset Color
    stdout.execute(ResetColor).unwrap();

    // Read user input
    let mut user_response: String = String::new();

    for line in stdin().lines() {
        match line {
            Ok(line) => user_response.push_str(&line),
            Err(e) => panic!("{}", e),
        }
    }

    // Trim whitespace and return
    return user_response.trim().to_string();
}
