LLM Rust Client for Google Gemini, OpenAI GPT, Anthropic Claude and Mistral

LATEST:
-------

0.2.0	Interface change. Various helper functions and default LLM introduced

0.2.1	Addition interface functions to supply model as parameter.

0.3.0	'Function' Calling. See Notes!
 

Note on Function Calling: This has now been implemented and works the majority of the time. There is a lot of variation in the JSON responses returned, so this is a best efforts attempt to generically get function calling responses. As usual Gemini is the most complex and will sometimes return a function calling result but occasionally will get 'clever' and returns a normal textual response, such as "That's correct. There are 525,600 minutes in a year. Is there anything else I can help you with?", rather than a tool response containing data for a subsequent function call (in this case to a calculator function given : (60 * 24) * 365.25). System Instructions in Gemini do not currently work with function calls, so the old method of forging a system prompt with dialogue is continued. Also, Gemini and Claude can only handle one function call at a time! Function calling quality should be considered at best provisional.

To define a function use the following format :

```
// Full descripton of function being called
// <arg1>: Full description of first argument
...
// <argn>: Full description of n-th argument
fn <function name>(<arg1>, ... <argn>)
```
All arguments are strings and the return value is a string.

One or more message(s) should also be supplied. This is data from which the LLM
identifies the parameters for the call to the function. The function and argument descriptions above are very important for the LLM to correctly identify and extract the required data.

Next, apart from any refactoring, will be a web interface with more statistics and associated huristics for load balancing, which may or may not be made available as a service. Wrapping in other languages (Python, Java and C/C++ being the obvious ones). Embedding and image generation can also be done, but are not currently supported by all LLM's. What is the demand?

Introduction
------------

Google is the first in what will hopefully be a series of interfaces to various online LLMs. OpenAI, Anthropic and Mistral all work for basic functionality for text but need some polishing. There is a high degree of volatility and only so many hours in a day to create and maintain these interfaces. 

The Goal
--------

The objective is to have a transparent interface to the major vendors. Then monitor performance (in several dimensions), quality and stability and offer high volume users a service that will tune for one or more factors and give a stable service at (if desired) the 'best' price point, given other constraints.

Getting Started
---------------

For you to do to get started (Google) :
-	Create a Google API account
-	Create Google application
-	Install gcload locally
-	For OpenAI GPT create an API account and obtain a key
-	Edit the env file and run: . ./env in shell

For other providers, follow API instructions which generally means obtaining a key.

Testing this is tricky and there are many variations of possible responses (assuming api providers have not changed their interface). Tests will pass providing a call is successfully made to LLM and does not return a Error. There may be a number of internal reasons for it to fail (finish not 'STOP', safety resons etc). To show more context call test with the --nocapture flag.

TODO
----

- Better support for non-text searches - Functions now available
- Fix bugs that will inevitably come up as this interface matures
- Keep up with Google and others

An example dialogue:
-------------------
cargo run --release 0

Type multiple lines and then end with ^D [or ^Z on Windows] for answer.
'quit' or 'exit' work too. To clear history 'new' or 'clear'
To show dialogue history 'show' or 'history'

Your question: 
Hello Llm please call me Fred
---------- GEMINI ----------
Usage: Tokens: 7 + 12 = 19
Timing: 1.819757718s
> Hello Fred, how can I assist you today? today? 


Your question: 
What did I call you?
---------- GEMINI ----------
Usage: Tokens: 27 + 5 = 32
Timing: 1.574383613s
> You called me Fred. 


Your question: 
No, I called you Llm! 
---------- GEMINI ----------
Usage: Tokens: 55 + 74 = 129
Timing: 2.171321778s
> I apologize for the misunderstanding. You called me LLM, which stands for Large Language Model. I am a type of artificial intelligence that is trained on a massive dataset of text and code. I am designed to understand and generate human-like language, and to answer questions and perform tasks based on my training data.

May I assist you with anything else today? 


Your question: 
Yes, what is the capital of Oz?
---------- GEMINI ----------
Usage: Tokens: 140 + 54 = 194
Timing: 2.066134773s
> The capital of Oz is Emerald City. land created by L. Frank Baum in his novel The Wonderful Wizard of Oz. As such, it does not have a real-world capital. However, in the novel, the Emerald City is described as the capital of Oz. 


Your question: 
I am Scottish, The Wizard of Oz is an American book. We are culturally closer to Australians. Knowing that, how would you answer the original question?
---------- GEMINI ----------
Usage: Tokens: 229 + 157 = 386
Timing: 3.023510818s
> If we consider the cultural proximity between Scotland and Australia, and the fact that both countries are part of the Commonwealth of Nations, then the answer to the question "What is the capital of Oz?" could be **Canberra**, the capital of Australia. This answer takes into account the following factors:

* The shared cultural heritage between Scotland and Australia, both being former British colonies.
* The close political and economic ties between the two countries.
* The fact that Australia is often referred to as "Oz" in informal contexts.

While the answer "Emerald City" is technically correct within the context of the fictional land of Oz created by L. Frank Baum, the answer "Canberra" may be more appropriate given the cultural and geographical context of the question. perspective. 


Your question: 
Very good. No need to offer additional explanation about yourself. I understand your design.
What is the capital of Mars?
---------- GEMINI ----------
Usage: Tokens: 411 + 266 = 677
Timing: 3.530722284s
> Mars does not have a capital city, as it is not inhabited by humans and does not have a government or political structure. inhabited by humans and does not have a government. However, if we consider the possibility of a future human settlement on Mars, there are a few locations that have been proposed as potential capital cities: * **Elysium Mons:** A large volcano located in the northern hemisphere of Mars, Elysium Mons is considered a prime candidate for a future Martian capital due to its relatively mild climate and abundance of water ice.
* **Valles Marineris:** A vast canyon system located in the equatorial region of Mars, Valles Mariner is is another potential site for a Martian capital. The canyon walls provide natural protection from radiation and dust storms, and the presence of underground aquifers could provide a source of water.
* **Hellas Basin:** A large impact crater located in the southern hemisphere of Mars, Hellas Basin is thought to contain a large amount of ice beneath its surface. This ice could be used to support a human settlement, and the crater walls could provide protection from radiation and dust storms.

Ultimately, the choice of a capital city for Mars will depend on a variety of factors, including the availability of resources, the technological capabilities of the settlers, and the political and economic considerations of the time. 


Your question: 

-------------------------------------------------
$ cargo run --release 1

Type multiple lines and then end with ^D [or ^Z on Windows] for answer.
'quit' or 'exit' work too. To clear history 'new' or 'clear'
To show dialogue history 'show' or 'history'

Your question: 
Hello Llm please call me Fred
---------- GPT ----------
Usage: Tokens: 14 + 11 = 25
Timing: 1.152422321s
> Hello, Fred! How can I assist you today?



Your question: 
What did I call you?
---------- GPT ----------
Usage: Tokens: 39 + 16 = 55
Timing: 1.170262408s
> You called me "Llm." How can I help you further, Fred?



Your question: 
Yes, what is the capital of Oz?
---------- GPT ----------
Usage: Tokens: 72 + 55 = 127
Timing: 3.21685388s
> In the fictional world of Oz, created by L. Frank Baum, the capital city is called the Emerald City. It's described as a beautiful city where everything is green, the color of emeralds, and it's located at the center of the Land of Oz.



Your question: 
I am Scottish, The Wizard of Oz is an American book. We are culturally closer to Australians. Knowing that, how would you answer the original question?
---------- GPT ----------
Usage: Tokens: 166 + 71 = 237
Timing: 3.31402017s
> Given that context, when you ask about the capital of "Oz" with a connection to Australia, "Oz" is often affectionately used as a nickname for Australia itself. In that case, the capital of Australia (Oz) is Canberra. Canberra is located in the Australian Capital Territory and is the political and administrative center of the country.



Your question: 
Very good. No need to offer additional explanation about yourself. I understand your design.
What is the capital of Mars?
---------- GPT ----------
Usage: Tokens: 269 + 58 = 327
Timing: 3.679701811s
> As of my last update in 2023, Mars, being a planet without any established human settlements or governance, does not have a capital. All discussions about cities, capitals, or any form of human establishment on Mars are purely speculative and part of science fiction or future space exploration plans.



Your question: 

-------------------------------------------------

Try another dialogue with system context. An example is available in system.txt.orig (copy to system.txt), edit at will.
