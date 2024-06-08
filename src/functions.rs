use std::collections::HashMap;
use serde_json::Value;
use serde_json::Value::*;
use std::string::String;
use regex::Regex;
use serde_derive::{Serialize, Deserialize};
use peg::*;
use peg::error::ParseError;
use peg::str::LineCol;
use stemplate::*;
use crate::common::{LlmType, LlmReturn};
use crate::caller::call_my_functions;

// Internal functions for parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseFunction {
    pub function: String,
    pub arguments: Vec<ParseArgument>,
}

impl ParseFunction {
    fn new(function: &str, arguments: Vec<ParseArgument>) -> Self {
        ParseFunction { function: function.to_string(), arguments }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseArgument {
    pub name: String,
    pub desc: String,
}

impl ParseArgument {
    fn new(name: &str, desc: &str) -> Self {
        ParseArgument { name: name.to_string(), desc: desc.to_string() }
    }
}

/// Wrapper used by GPT, Mistral and Groq
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub r#type: String,
    pub function: Function,
}

impl FunctionCall {
    pub fn functions(function: Option<Vec<Function>>) -> Vec<Self> {
        match function {
            None => {
               vec![]
            },
            Some(functions) =>  {
                functions.iter()
                    .map(|f| 
                        FunctionCall {
                            r#type: "function".to_string(),
                            function: f.clone()
                        }
                    ).collect()
                }
        }
    }
}

/// used by Gemini
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    pub function_declarations: Function,
}

impl FunctionDeclaration {
    pub fn functions(function: Option<Vec<Function>>) -> Vec<Self> {
        match function {
            None => {
               vec![]
            },
            Some(functions) =>  {
                functions.iter()
                    .map(|f| 
                        FunctionDeclaration {
                            function_declarations: f.clone()
                        }
                    ).collect()
                }
        }
    }
}

// No wrapper for Claude

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Function {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Parameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Parameters>,
}

impl Function {
    pub fn new(name: &str, description: &str, parameters: Parameters, is_gpt: bool) -> Self {
        Function {
            name: name.to_string(),
            description: description.to_string(),
            parameters: if is_gpt { Some(parameters.clone()) } else { None },
            input_schema: if !is_gpt { Some(parameters.clone()) } else { None },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Parameters {
    pub r#type: String,
    pub properties: Properties,
    pub required: Vec<String>
}

impl Parameters {
    pub fn new(ptype: &str, properties: Properties, required: Vec<String>) -> Self {
        Parameters {
            r#type: ptype.to_string(),
            properties,
            required
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Properties {
    //#[serde(rename = "function")]
    //pub function: Vec<ParameterType>
    #[serde(flatten)]
    parameter_name: HashMap<String, ParameterType>
}

impl Properties {
    pub fn new(name: &str, parameter_type: ParameterType) -> Self {
        let mut pt = HashMap::new();

        pt.insert(name.to_string(), parameter_type);

        Properties {
            parameter_name: pt
        }
    }

    pub fn new_type(name: &str, ptype: &str, pdesc: &str) -> Self {
        let mut pt = HashMap::new();

        pt.insert(name.to_string(), ParameterType::new(ptype, pdesc));

        Properties {
            parameter_name: pt
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParameterType {
    pub r#type: String,
    //pub r#enum: String,
    pub description: String,
}

impl ParameterType {
    pub fn new(ptype: &str, description: &str) -> Self {
        ParameterType {
            r#type: ptype.to_string(),
            description: description.to_string()
        }
    }
}

pub fn json_function(provider: &str, func_defs: &[&str]) -> Result<String, ParseError<LineCol>> {
/*
    let func = match provider {
        "gpt" | "mistral" | "groq" => 
r#"
"function": {
    "name": "${func}",
    "description": "${func_desc}",
    "parameters": {
        "type": "object",
        "properties": {
        ${*,all_args}
        },
        "required": [${*,mand_args}]
    }
}
"#,
        "claude" => 
r#"
"name": "${func}",
"description": "${func_desc}",
"input_schema": {
  "type": "object",
  "properties": {
  ${*,all_args}
  },
  "required": [${*,mand_args}]
}
"#,
        "_gemini" => 
r#"
"type": "function",
"function": {
    "name": "${func}",
    "description": "${func_desc}",
    "parameters": {
        "type": "object",
        "properties": {
        ${*,all_args}
        },
        "required": [${*,mand_args}]
    }
}
"#,
        _ => todo!(),
   };
*/

    let func = match provider {
        "claude" => r#"
"name": "${func}",
"description": "${func_desc}",
"input_schema": {
  "type": "object",
  "properties": {
  ${*,all_args}
  },
  "required": [${*,mand_args}]
}
"#,
        _ => r#"
"name": "${func}",
"description": "${func_desc}",
"parameters": {
  "type": "object",
  "properties": {
  ${*,all_args}
  },
  "required": [${*,mand_args}]
}
"#,
    };

    let all_args = r#"
"${arg}": {
  "type": "string",
  "description": "${arg_desc}"
}
"#;

    let mand_args = r#""${marg}""#;

//println!("func_defs: {:?}", llmfunc::func(func_defs[0], func, all_args, mand_args));
//println!("func_defs: {func_defs:?}");
    let defs: Vec<String> = func_defs.iter()
        .flat_map(|f| {
            llmfunc::func(f, func, all_args, mand_args)
        })
        .map(|f| {
            format!("{{ {f} }}")
        })
        .collect();
//println!("defs: {defs:?}");

    Ok(format!("[ {} ]", defs.join(",")))
    //Ok(defs.join(","))
}

pub fn unpack_functions(h: HashMap<String, Vec<String>>) -> Option<Vec<ParseFunction>> {
    let func = h.get("func");
    let args = h.get("args");

    if let Some(func) = func {
        if let Some(args) = args {
            let funcs = func.iter().zip(args.iter())
                .map(|(f, a)| {
                    // Claude has expression wrapped in String(..)
                    let a = {
                        let re = Regex::new(r"(.*)String\((.*)\)(.*)").unwrap();
                        match re.captures(a) {
                            Some(bits) => format!("{}{}{}", &bits[1], &bits[2], &bits[3]),
                            None => a.to_string()
                        }
                    };
//println!("{f}: {a} - {}", a.contains("String"));
                    if a.starts_with('{') && a.ends_with('}') {
                        let fh: Result<HashMap<String, String>, _> = serde_json::from_str(&a);
                        if let Ok(fh) = fh {
                            let args: Vec<ParseArgument> = fh.iter()
                                .map(|(pn, pv)| ParseArgument::new(pn, pv))
                                .collect();

                            ParseFunction::new(f, args)
                        } else {
                            ParseFunction::new(f, vec![])
                        }
                    } else {
                        ParseFunction::new(f, vec![])
                    }
                })
                .collect();

            return Some(funcs);
        }
    }

    None
}

pub fn get_functions(val: &Value, found: &Vec<String>) -> HashMap<String, Vec<String>> {
    fn getter(val: &Value, places: &str, found: &mut HashMap<String, Vec<String>>) -> HashMap<String, Vec<String>> {
        let mut v = val;
        let items: Vec<&str> = places.split(':').collect();
        let var = *items.last().unwrap();

        for (pos, i) in items.iter().enumerate() {
            if *i == var {
                if let Value::String(it) = v {
                    let key = &i[2..var.len()-1];

                    found.entry(key.to_string())
                        .and_modify(|v| v.push(it.to_string()))
                        .or_insert(vec![it.to_string()]);
                } else if let Value::Number(it) = v {
                    let key = &i[2..var.len()-1];

                    found.entry(key.to_string())
                        .and_modify(|v| v.push(it.to_string()))
                        .or_insert(vec![it.to_string()]);
                } else if let Value::Object(it) = v {
                    let key = &i[2..var.len()-1];

                    found.entry(key.to_string())
                        .and_modify(|v| v.push(format!("{it:?}")))
                        .or_insert(vec![format!("{it:?}")]);
                }
            } else if let Value::Array(it) = v {
                for a in it {
                    getter(a, &items[pos..].join(":"), found);
                }
            } else {
                v = &v[i];
            }
        }

        found.clone()
    }

    let mut res: HashMap<String, Vec<String>> = HashMap::new();

    for i in found {
        getter(val, i, &mut res);
    }

    res
}

pub fn find_function(v: &Value) -> Vec<String> {
    fn finder(v: &Value, res: String, found: &mut Vec<String>) -> Vec<String> {
        match v {
            Null => {
            },
            Bool(_b) => {
            },
            Number(_n) => {
            },
            String(s) => {
                let re = Regex::new(r#"\$\{[A-Za-z0-9_]+\}"#).unwrap();
                if re.is_match(s) {
                    let f = format!("{res}{s}");

                    if !found.contains(&f) {
                        found.push(f);
                    }
                }
            },
            Array(a) => {
                for v in a {
                    finder(v, res.clone(), found);
                }
            },
            Object(o) => {
                for (k, v) in o.iter() {
                    finder(v, res.clone() + k + ":", found);
                }
            },
        };

        found.clone()
    }

    finder(v, String::new(), &mut vec![])
}

peg::parser!( grammar llmfunc() for str {
    pub rule func(func: &str, all_args: &str, mand_args: &str) -> String
        = "\n"* fc:func_comment()+ ac:arg_comment()+ "fn"? _ f:ident() _ "(" a:arg_ident() ** comma() ")" _ "\n"* _ {
            let cnt = ac.iter().enumerate()
                .filter(|(i, arg)| {
                    arg.starts_with(a[*i]) || format!("*{arg}").starts_with(a[*i])
                }).count();
            if ac.len() == a.len() && a.len() == cnt {
                let ma: Vec<&str> = a.iter()
                    .filter(|a| !a.starts_with('*'))
                    .map(|a| &a[..])
                    .collect();
                let a: Vec<&str> = a.iter()
                    .map(|a| if let Some(stripped) = a.strip_prefix('*') { stripped } else { a })
                    .collect();
                let mut h: HashMap<&str, String> = HashMap::new();

                h.insert("func", f.to_string());
                h.insert("func_desc", fc[0].to_string());
                h.insert("arg", a.join("|"));
                h.insert("arg_desc", ac.join("|"));
                h.insert("marg", ma.join("|"));
                h.insert("all_args", all_args.to_string());
                h.insert("mand_args", mand_args.to_string());
                h.insert("func_call", func.to_string());

                Template::new("${func_call}").render(&h)
            } else {
                "Error: Argument names do not match".to_string()
            }
        }

    rule _ = [' ']*

    rule comma() = "," " "* 

    rule ident() -> &'input str
        = s:$(['a'..='z'|'A'..='Z'|'0'..='9'|'_']+) { s }

    rule arg_ident() -> &'input str
        = s:$("*"? ident()) { s }

    rule func_comment() -> &'input str
        = _ "/"*<2,3> _ s:$(comment()) _ "\n"+ { s }

    rule arg_comment() -> &'input str
        = _ "/"*<2,3> _ a:$(ident() ":" comment()) _ "\n"+ { a }

    rule comment() -> &'input str
        = s:$([';'..='`'|'a'..='~'|'_'|' '..='9']+) { s }
});

pub fn get_function_json(llm: &str, function: &[&str]) -> Option<Vec<Function>> {
    let func = match json_function(llm, function) {
        Ok(res) => res,
        Err(_) => {
            eprintln!("{:?}: Invalid function definition", function);
            return None;
        }
    };

    serde_json::from_str(&func).ok()
}

pub fn call_actual_function(res: Option<LlmReturn>) -> Vec<String> {
    if let Some(llm_return) = res {
        match llm_return.llm_type {
            LlmType::GEMINI_TOOLS |
            LlmType::GPT_TOOLS |
            LlmType::CLAUDE_TOOLS |
            LlmType::MISTRAL_TOOLS |
            LlmType::GROQ_TOOLS => 
                call_my_functions(serde_json::from_str::<Vec<ParseFunction>>(&llm_return.text)),
            _ => vec!["LLM failed to treat query as a function call".to_string()]
        }
    } else {
        vec!["LLM returned unexpected JSON".to_string()]
    }
}
