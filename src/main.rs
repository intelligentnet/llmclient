use crossterm::{
    style::{Color, ResetColor, SetForegroundColor},
    ExecutableCommand,
};
use std::io::{stdin, stdout};
use llmclient::gemini::{call_gemini, Content};

#[tokio::main]
async fn main() {
    highlight("Type multiple lines and then end with ^D [or ^Z on Windows] for answer.");
    highlight("'quit' or 'exit' work too. To clear history 'new' or 'clear'");
    highlight("To show dialogue history 'show' or 'history'");

    let mut prompts: Vec<String> = Vec::new();

    // Are 'system' context instructions available?
    let system_data = std::fs::read_to_string("system.txt");
    if let Ok(ref system) = system_data {
        prompts.push(system.into());
        prompts.push("Understood".into());
    }

    loop {
        let prompt = get_user_response("Your question: ");

        if prompt.is_empty() || prompt.to_lowercase() == "quit" || prompt.to_lowercase() == "exit" {
            break;
        } else if prompt.to_lowercase() == "new" || prompt.to_lowercase() == "clear" {
            prompts.truncate(if system_data.is_ok() { 2 } else { 0 });

            continue;
        } else if prompt.to_lowercase() == "show" || prompt.to_lowercase() == "history" {
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
        Ok((text, finish_reason, citations, metadata)) => {
            if !finish_reason.is_empty() && finish_reason != "STOP" {
                println!("Finish Reason: {}", finish_reason);
            }
            if !citations.is_empty() {
                println!("{}", citations);
            }

            println!("{}", metadata);

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

fn highlight(text: &str) {
    let mut stdout: std::io::Stdout = stdout();

    // Print the question in a specific color
    stdout.execute(SetForegroundColor(Color::Blue)).unwrap();
    println!("{}", text);

    // Reset Color
    stdout.execute(ResetColor).unwrap();
}

// Get user request
fn get_user_response(question: &str) -> String {
    println!();
    highlight(question);

    // Read user input
    let mut user_response: String = String::new();

    for line in stdin().lines() {
        match line {
            Ok(line) => user_response.push_str(&line),
            Err(e) => panic!("{}", e),
        }
    }

    // Trim whitespace and return
    user_response.trim().to_string()
}
