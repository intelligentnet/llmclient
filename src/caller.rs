use crate::functions::*;
use evalexpr::eval;

/**
  This function should be implemented for the users functions.

  This is just an example.
**/
pub fn call_my_functions(js: Result<Vec<ParseFunction>, serde_json::Error>) -> Vec<String> {
    match js {
        Ok(funcs) => 
            funcs.into_iter()
                .map(|f| {
                    if f.function == "arithmetic" && f.arguments.len() == 1 {
                        let desc = eval(&f.arguments[0].desc);

                        if let Ok(v) = desc {
                            format!("{} -> {}", f.function, v)
                        } else {
                            format!("Invalid formula for: {}", f.function)
                        }
                    } else if f.function == "apple" && f.arguments.len() == 2 {
                        format!("You have found an apple({}, {})", f.arguments[0].desc, f.arguments[1].desc)
                    } else {
                        format!("No function found : {} {:?}", f.function, f.arguments)
                    }
                }
                )
                .collect(),
        Err(e) => 
            vec![format!("{e:?}")]
    }
}
