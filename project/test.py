import pandas as pd

data = {
    "text": [
        "I absolutely love this product! It’s amazing.",
        "The service was terrible, and I will not return.",
        "Great experience, highly recommend to everyone!",
        "Not worth the money, very disappointing.",
        "The quality exceeded my expectations, fantastic purchase.",
        "Worst experience ever, I regret buying this.",
        "Decent product, but the price is too high.",
        "The customer service was outstanding and very helpful.",
        "I wouldn’t recommend this to my friends.",
        "Excellent value for money, very satisfied!",
        "The delivery was quick, and the packaging was perfect.",
        "Horrible quality, broke within a week.",
        "This item is exactly what I needed, works perfectly.",
        "Not as described, very disappointed.",
        "Amazing customer service, resolved my issue quickly.",
        "I’ll never buy from this brand again.",
        "Product arrived damaged, very poor handling.",
        "Great value for the price, highly satisfied.",
        "I love the design, but the material feels cheap.",
        "The food was delicious and served hot.",
        "I waited over an hour, terrible service.",
        "A must-buy for anyone looking for quality.",
        "This is the worst product I’ve ever purchased.",
        "Fantastic experience, would definitely buy again.",
        "It didn’t meet my expectations, not recommended.",
        "Fast shipping and great communication from the seller.",
        "Overpriced for what it offers, not worth it.",
        "Exceeded my expectations, truly wonderful.",
        "The color faded after one wash, disappointing.",
        "Beautiful packaging, but the product is subpar.",
        "Highly functional and easy to use.",
        "Not user-friendly, too complicated to set up.",
        "Great sound quality, perfect for music lovers.",
        "The battery life is terrible, needs improvement.",
        "The app keeps crashing, very frustrating.",
        "Very responsive and professional customer support.",
        "Item is missing parts, very annoyed.",
        "The hotel staff was friendly and welcoming.",
        "The room was dirty, and the service was slow.",
        "This product changed my life for the better.",
        "Feels flimsy, not sure it will last long.",
        "Worth every penny, would recommend to anyone.",
        "The instructions were unclear, hard to assemble.",
        "Absolutely stunning, better than I expected.",
        "Looks good but doesn’t work as advertised.",
        "Incredible attention to detail, very satisfied.",
        "Doesn’t hold up well over time, poor durability.",
        "The flavors were bland, not enjoyable.",
        "This gadget makes my daily tasks so much easier.",
        "Arrived earlier than expected, very happy with it."
    ]
}

df = pd.DataFrame(data)
df.to_csv("sample_test_data_50.csv", index=False)

print("File 'testdata.csv' berhasil dibuat.")