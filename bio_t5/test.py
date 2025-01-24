import torch

def decode(preds):
    preds[preds == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    preds = [pred.strip() for pred in preds]
    return preds




batch = torch.load('one_batch.pt')
tokenizer = torch.load('tokenizer.pt')
model = torch.load('model.pt')


input_ids = batch['input_ids']
labels = batch['labels']
batch = batch.to('cuda')
res = model(**batch).logits

predictions = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=8,
                generation_config=model.generation_config,
            )

decoded_preds = decode(predictions)

# tokenizer.decode([x for x in predictions[0].tolist() if x != -100])


