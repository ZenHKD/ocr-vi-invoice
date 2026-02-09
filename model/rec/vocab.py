# Vietnamese Character Set
# This string contains all characters that the model should be able to recognize.

# Standard lowercase Vietnamese vowels with tones
vowels = "aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ"

# Consonants and other characters
consonants = "bcdđghklmnpqrstvxfjwz"

# Digits and symbols
digits = "0123456789"
# Currency symbols (for international invoices)
currency = "$€£¥₫"  # USD, EUR, GBP, JPY, VNĐ
symbols = "!\"#%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

# Full vocabulary
# Note: We include both lowercase and uppercase
vietnamese_vocab = vowels + vowels.upper() + consonants + consonants.upper() + digits + currency + symbols

# Export as a simple string
VOCAB = "".join(sorted(list(set(vietnamese_vocab))))
