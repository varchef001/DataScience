# Reflection

## 1. Tool description

The notes search description needs to explain that the tool matches a literal topic phrase, rather than understanding a whole question. A vague description such as “search notes” would not make that limitation clear. The implemented description specifies case-insensitive matching and returned filenames and line numbers. For example, “gradient descent” matches the sample notes, while a full question may not. The local tests confirm that the short phrase finds four lines. However, I have not measured a change in the model's behavior because no API key was available. A live before-and-after comparison is still needed to answer that part fully.

## 2. Wrong tool or answer

I could not honestly report an observed wrong choice by the live model. Instead, the offline tests deliberately sent invalid calculator input and a division-by-zero expression. Both were handled without crashing, and the error was passed back into the conversation. One likely model mistake is treating the percent sign as percentage notation, although the calculator uses it for remainder. The description therefore tells the model to use division by 100. The five live prompts should check whether that instruction works in practice.

## 3. More powerful tools

A delete-file or email tool could cause real damage if the model misunderstood a request or followed instructions hidden inside notes. I would require a preview and explicit user confirmation before either action. I would also restrict file access to an approved folder and verify email recipients. The eight-step limit prevents endless looping, but it cannot undo a harmful action, so confirmation should happen before execution.
