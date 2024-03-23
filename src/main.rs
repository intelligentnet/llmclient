use crossterm::{
    style::{Color, ResetColor, SetForegroundColor},
    ExecutableCommand,
};
use std::io::{stdin, stdout};
use llmclient::gemini::{call_gemini, Content};
use llmclient::gpt::{call_gpt, GptMessage};

#[tokio::main]
async fn main() {
    let mut llm = 0;
    let args: Vec<String> = std::env::args().collect();

    if args.len() <= 1 {
        highlight("Please supply 1 argument to indicate the LLM to run : 0 = gemini, 1 = gpt");
        highlight("This run will default to gemini\n\n");
    } else {
        llm = args[1].parse::<usize>().unwrap_or_default();
    }

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

        let res: Result<String, Box<dyn std::error::Error + Send>> = llm_chat(llm, &prompts[..]).await;

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

async fn gemini(content: Vec<Content>) -> String {
    match call_gemini(content).await {
        Ok(ret) => ret.to_string(),
        Err(e) => e.to_string()
    }
}

async fn gpt(messages: Vec<GptMessage>) -> String {
    match call_gpt(messages).await {
        Ok(ret) => ret.to_string(),
        Err(e) => e.to_string()
    }
}

async fn llm_chat(llm: usize, prompts: &[String]) -> Result<String, Box<dyn std::error::Error + Send>> {
    let ret =
        match llm {
            0 => {
                let content = Content::dialogue(prompts);

                gemini(content).await
            },
            1 => {
                let message = GptMessage::dialogue(prompts);

                gpt(message).await
            },
            _ => "No such llm".to_string()
        };

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
