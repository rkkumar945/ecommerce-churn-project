"""Transaction cleaning for Online Retail II (Days 31-35)."""

import pandas as pd

REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

# Postage / fees / adjustments / test rows - not real products.
# (Original list + codes actually present in the data: C2 carriage, BANK CHARGES, PADS, D, TEST*)
NON_PRODUCT_CODES = [
    "POST",
    "DOT",
    "M",
    "ADJUST",
    "GIFT",
    "AMAZONFEE",
    "BANK CHARGES",
    "C2",
    "PADS",
    "D",
    "TEST001",
    "TEST002",
]


def non_product_mask(stockcodes) -> pd.Series:
    """Shared non-product definition used by both cleaning and feature building.

    This is intentionally explicit and opt-in; changing the list changes customer eligibility,
    labels and features, so retraining is required before shipping a new policy.
    """
    codes = {str(code).upper() for code in NON_PRODUCT_CODES}
    return stockcodes.astype(str).str.upper().isin(codes)

def clean_transactions(
    df: pd.DataFrame,
    exclude_non_product: bool = False, #dataset me POST,ADJUST,BANK CHARGES etc. haise entries bhi hoti hai,inme paisa to add ho jata hai per te sale nahi hue.
    drop_nonpositive_price: bool = False, #agar koi item ka price 0 ya negative hai to ye valus model ko confuse kar degi.
) -> pd.DataFrame:
    """Drop guest checkouts, fill Description, drop duplicates and cancellations, fix dtypes.

    exclude_non_product:    True -> POST/DOT/M/ADJUST/... rows hata do (README mein "excluded" likha hai).
    drop_nonpositive_price: True -> Price <= 0 wali rows (free/adjustment lines) hata do.
    Dono default False hain taaki purane notebook results (aur README ke numbers) reproduce hon.
    Input DataFrame change nahi hota.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns] #ye check karta hai ki data me sare jaruri columns hai ya nahi , agar nahi to error msg dega(kon sa column missing hai)., agar columns nahi hue to cleaning steps kaam nahi karenge or code crash hoga.
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

          #
    df = df.dropna(subset=["Customer ID"]).copy() #jin rows me customer ID nahi hai(NaN/null) unhe hata deta hai or new copy ban deta hai.
    #Jab hum Pandas mein kisi DataFrame ka slice ya filter banate hain, toh Python kabhi-kabhi naye DataFrame ko original wale ka view (reference) hi maan leta hai. Aise mein agar hum koi naya column ya modification karein, toh Pandas ek SettingWithCopyWarning deta hai, .copy() lagane se Pandas ko saaf pata chal jata hai ki hum ek bilkul nayi, independent copy par kaam kar rahe hain taaki original data disturb na ho.
    df["Description"] = df["Description"].fillna("Unknown Product")#jaha product ka discription missing(NaN) hai waha "Unknown Product" likh dega.
    df = df.drop_duplicates() #duplicates rows ko hata deta hai, iski wajha se sales number double count ho jate hai.
    df = df[
        ~df["Invoice"].astype(str).str.startswith("C") # (~) un invoices ko hata deti hai jinke starting me C aata hai, warna return orders bhi valid sales ki tarah count ho jainge jis se revenue galat ho jaiga.
    ].copy()  # cancellations/returns
    if exclude_non_product:
        df = df[~non_product_mask(df["StockCode"])].copy()
    if drop_nonpositive_price:      # 0 ya negative price wali rows ko hata deta hai, 
        df = df[df["Price"] > 0].copy() 
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]) # InvoiceDate column ko string se pandas ke datetime format me convert karta hai. isi se hum month,time,dates per based filtering,sorting,time-based grouping kar sakte hai.
    df["Customer ID"] = df["Customer ID"].astype(int) # covert CustomerID into integer format.
    return df
