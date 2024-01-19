import argparse
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


first_model_name = "masakhane/m2m100_418M_yor_en_rel_ft"
first_model_token =  M2M100Tokenizer.from_pretrained(first_model_name)
first_model = M2M100ForConditionalGeneration.from_pretrained(first_model_name)

second_model_name = "masakhane/m2m100_418M_en_yor_rel_ft"
second_model_token =  M2M100Tokenizer.from_pretrained(second_model_name)
second_model =  M2M100ForConditionalGeneration.from_pretrained(second_model_name)

original_texts = "Bawo ni, ọmọ mi?","Epo rọ̀bì tó ń bọ̀ lágbègbè náà ti fẹ́rẹ̀ẹ́ parí","Ibo lẹ lọ?"
original_texts

def format_batch_texts(batch_texts):
    formated_batch = ["{}".format( text) for text in batch_texts]
    return formated_batch
format_batch_texts(original_texts)

def perform_translation(batch_texts, model, tokenizer, language="en"):
    formated_batch_texts = format_batch_texts( batch_texts)
    model_inputs = tokenizer(formated_batch_texts, return_tensors="pt", padding = True, truncation = True)
    translated = model.generate(**model_inputs)
    translated_texts = {tokenizer.decode(t, skip_special_tokens = True) for t in translated}
    return translated_texts

translated_texts = perform_translation(original_texts, first_model, first_model_token)
translated_texts
back_translated_texts = perform_translation(translated_texts, second_model, second_model_token)
back_translated_texts

def augmented_data(original_texts, back_translated_batch):
    return set(original_texts) | set(back_translated_batch)

def perform_back_translation(batch_texts, original_language = "yor", temporary_language = "en"):
    temp_translated_batch = perform_translation(batch_texts, first_model, first_model_token, temporary_language)
    back_translated_batch = perform_translation(temp_translated_batch, second_model, second_model_token, original_language)
    return augmented_data(original_texts, back_translated_batch)

final_augmented_data =  perform_back_translation(original_texts)
final_augmented_data

cmd_args = ["--model_name", "masakhane/m2m100_418M_yor_en_rel_ft",
            "--original_texts", "Bawo ni, ọmọ mi?", "Epo rọ̀bì tó ń bọ̀ lágbègbè náà ti fẹ́rẹ̀ẹ́ parí", "Ibo lẹ lọ?",
             "--tokenizer_name", "masakhane/m2m100_418M_yor_en_rel_ft"
            ]


# cmd_args = [ --model_name "_model",
#                       --original_texts_file "path-to-original_texts.txt",
#                       --back_translation_language "en",
#                       

parser = argparse.ArgumentParser(description="Perform back-translation with M2M100 models.")
parser.add_argument("--model_name", type=str, required=True, help="Model name for tokenizer and model loading.")
#parser.add_argument("--original_texts_file", type=str, required=True, help="Path to a file containing original texts for translation.")
parser.add_argument("--original_texts", nargs="+", required=True, help="Path to a file containing original texts for translation.")
parser.add_argument("--back_translation_language", type=str, default="en", help="Language for back translation.")
parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name for loading.")
args = parser.parse_args(cmd_args)

# with open(args.original_texts_file, 'r', encoding='utf-8') as file:
    # original_texts = [line.strip() for line in file]


tokenizer = M2M100Tokenizer.from_pretrained(args.model_name)
model = M2M100ForConditionalGeneration.from_pretrained(args.model_name)

original_texts = args.original_texts
translated_texts = perform_translation(original_texts, model, tokenizer)
back_translated_texts = perform_translation(translated_texts, model, tokenizer, language=args.back_translation_language)
final_augmented_data =  perform_back_translation(original_texts)

print("Original Texts:", original_texts)
print("Final Augmented Data:", final_augmented_data)