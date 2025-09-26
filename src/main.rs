use crossterm::{
    style::{Color, ResetColor, SetForegroundColor},
    ExecutableCommand,
};
use std::io::{stdin, stdout};
use llmclient::common::call_llm_model;

#[tokio::main]
async fn main() {
    let mut llm = "4";
    let args: Vec<String> = std::env::args().collect();

    if args.len() <= 1 {
        highlight("Please supply first argument to indicate the LLM to run : 0 = Gemini, 1 = GPT, Claude = 2, Mistral = 3, Deepseek = 4, Groq = 5");
        highlight(&format!("This run will default to {llm}\n"));
    } else {
        llm = &args[1];
    }

    let mut model: String = 
         match llm {
            "0" | "gemini" =>
                std::env::var("GEMINI_MODEL").expect("GEMINI_MODEL not found in enviroment variables"),
            "1" | "gpt" | "openai" =>
                std::env::var("GPT_MODEL").expect("GPT_MODEL not found in enviroment variables"),
            "2" | "claude" =>
                std::env::var("CLAUDE_MODEL").expect("CLAUDE_MODEL not found in enviroment variables"),
            "3" | "mistral" =>
                std::env::var("MISTRAL_MODEL").expect("MISTRAL_MODEL not found in enviroment variables"),
            "4" | "deepseek" =>
                std::env::var("DEEPSEEK_MODEL").expect("DEEPSEEK_MODEL not found in enviroment variables"),
            "5" | "groq" =>
                std::env::var("GROQ_MODEL").expect("GROQ_MODEL not found in enviroment variables"),
            _ => std::env::var("GROQ_MODEL").expect("GROQ_MODEL not found in enviroment variables"),
        };

    if args.len() <= 2 {
        highlight("Please supply argument to indicate the model to run");
        highlight(&format!("This run will default to {model}\n"));
        highlight("Please ensure model exists and is compatable with llm\n");
    } else {
        model = args[2].clone();
    }

    let model = &model;

    highlight(&format!("Running {llm}: {model}\n"));
    highlight("Type multiple lines and then end with ^D [or ^Z on Windows] for answer.");
    highlight("'quit' or 'exit' work too. To clear history 'new' or 'clear'");
    highlight("To show dialogue history 'show' or 'history'");
    highlight("To show optional system content 'system'");

    // Are 'system' context instructions available?
    let system_data = std::fs::read_to_string("system.txt");

    let system: String =
        if let Ok(system) = system_data {
            system
        } else {
            "".into()
        };

    let mut prompts: Vec<String> = Vec::new();

    // Statistics
    let mut timer = 0.0;
    let mut in_tok = 0;
    let mut out_tok = 0;
    let mut all_tok = 0;

    loop {
        let prompt = get_user_response("Your question: ");

        if prompt.is_empty() {
            break
        }

        let prompt_lower: &str = &prompt.to_lowercase();

        let prompt =
            match prompt_lower {
                "quit" | "exit" => {
                    break
                },
                "new" | "clear" => {
                    prompts.truncate(0);

                    continue
                },
                "show" | "history" => {
                    println!("{:?}", prompts);

                    continue
                },
                "system" => {
                    println!("{:?}", system);

                    continue;
                },
                _ => prompt,
            };

        prompts.push(prompt);

        let res = match llm {
            "0" | "gemini" =>
                call_llm_model("gemini", model, &system, &prompts, 0.2, false, true).await,
            "1" | "gpt" | "openai" =>
                call_llm_model("gpt", model, &system, &prompts, 0.2, false, true).await,
            "2" | "claude" =>
                call_llm_model("claude", model, &system, &prompts, 0.2, false, true).await,
            "3" | "mistral" =>
                call_llm_model("mistral", model, &system, &prompts, 0.2, false, true).await,
            "4" | "deepseek" =>
                call_llm_model("deepseek", model, &system, &prompts, 0.2, false, true).await,
            "5" | "groq" =>
                call_llm_model("groq", model, &system, &prompts, 0.2, false, true).await,
            _ => todo!()
        };

        match res {
            Ok(ret) => {

                timer += ret.timing;
                in_tok += ret.usage.0;
                out_tok += ret.usage.1;
                all_tok += ret.usage.2;

                let ret = ret.to_string();
                println!("> {}", ret);

                prompts.push(ret);
            },
            Err(e) => {
                println!("Error (aborting): {}", e);

                break;
            }
        }
    }

    println!("Statistics: Elapsed time: {} secs, Tokens in: {} out: {} all: {}",
             timer, in_tok, out_tok, all_tok);
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
