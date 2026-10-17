from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Coffee Shop Sales", layout="wide")
st.title("Dashboard Coffee Shop Sales")
st.caption("Januari-Juni 2023. Mata uang belum dinyatakan pada file.")
folder = Path(__file__).resolve().parent
file_data = folder / "data_prepared" / "coffee_sales_prepared.csv"


@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)
    data["transaction_date"] = pd.to_datetime(data["transaction_date"])
    return data


if not file_data.exists():
    st.error("Jalankan audit. File coffee_sales_prepared.csv belum ada.")
    st.stop()

df = load_data(str(file_data), file_data.stat().st_mtime_ns)

# Sidebar Filter: Gerai dan Periode Bulan
st.sidebar.header("Filter data")
stores = sorted(df["store_location"].unique().tolist())
selected = st.sidebar.multiselect("Pilih gerai", stores, default=stores)
if not selected:
    st.info("Pilih minimal satu gerai.")
    st.stop()

data_filter = df[df["store_location"].isin(selected)].copy()

# Bulan selalu mengikuti cakupan sumber, termasuk bulan tanpa catatan.
months = pd.period_range(
    df["transaction_date"].min().to_period("M"),
    df["transaction_date"].max().to_period("M"), freq="M"
).astype(str).tolist()

monthly = data_filter.groupby("month").agg(
    total_sales=("sales", "sum"),
    total_transactions=("transaction_id", "nunique"),
    total_units=("transaction_qty", "sum")
).reindex(months)

previous = monthly["total_sales"].shift(1)

monthly["growth_pct"] = (
    (monthly["total_sales"] - previous) / previous * 100
).where(previous.notna() & previous.ne(0))
monthly["average_transaction_value"] = (
    monthly["total_sales"] / monthly["total_transactions"]
)
monthly = monthly.reset_index().rename(columns={"index": "month"})

# [Ide 3] Filter Periode Bulan di Sidebar
month_options = ["Semua Bulan"] + months
selected_month = st.sidebar.selectbox("Pilih bulan", month_options, index=0)

if selected_month == "Semua Bulan":
    data_display = data_filter.copy()
    st.sidebar.caption("Semua grafik mengikuti pilihan gerai (Januari - Juni 2023).")
else:
    data_display = data_filter[data_filter["month"] == selected_month].copy()
    st.sidebar.caption(f"Semua grafik difilter untuk gerai terpilih pada bulan {selected_month}.")


def format_angka(value, decimals=2):
    return (f"{value:,.{decimals}f}".replace(",", "_")
            .replace(".", ",").replace("_", "."))


# Perhitungan Kartu Metrik
if selected_month == "Semua Bulan":
    total_sales = data_filter["sales"].sum()
    total_transactions = data_filter["transaction_id"].nunique()
    average = total_sales / total_transactions if total_transactions else None
    growth = monthly["growth_pct"].iloc[-1]
    last_month = pd.Period(months[-1], freq="M")
    prev_month_label = str(last_month - 1)
    current_month_label = str(last_month)
    caption_scope = f"Total, transaksi, dan rata-rata: seluruh periode. Pertumbuhan: {last_month} terhadap {last_month - 1}."
else:
    total_sales = data_display["sales"].sum()
    total_transactions = data_display["transaction_id"].nunique()
    average = total_sales / total_transactions if total_transactions else None
    m_idx = months.index(selected_month)
    current_month_label = selected_month
    if m_idx > 0:
        prev_month_label = months[m_idx - 1]
        growth = monthly.loc[monthly["month"] == selected_month, "growth_pct"].values[0]
    else:
        prev_month_label = "bulan sebelumnya"
        growth = None
    caption_scope = f"Metrik untuk bulan {selected_month}. Pertumbuhan dihitung terhadap {prev_month_label}."

st.subheader("Ringkasan penjualan")
st.caption(caption_scope)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total penjualan", format_angka(total_sales))
col2.metric("Transaksi tercatat", format_angka(total_transactions, 0))
col3.metric("Rata-rata nilai transaksi",
            format_angka(average) if average is not None else "Belum tersedia")
col4.metric(f"Pertumbuhan {current_month_label} terhadap {prev_month_label}",
            f"{format_angka(growth)}%" if pd.notna(growth)
            else "Belum tersedia")

# Grafik Tren Penjualan Bulanan
st.subheader("Tren penjualan bulanan")
fig = px.line(monthly, x="month", y="total_sales", markers=True,
              labels={"month": "Bulan", "total_sales": "Nilai penjualan"})
fig.update_traces(line_color="#087F8C", connectgaps=False)
fig.update_layout(xaxis_type="category", yaxis_rangemode="tozero")
st.plotly_chart(fig, use_container_width=True, key="trend")

with st.expander("Lihat angka ringkasan bulanan"):
    table = monthly.rename(columns={
        "month": "Bulan", "total_sales": "Total penjualan",
        "total_transactions": "Transaksi tercatat",
        "total_units": "Unit terjual", "growth_pct": "Pertumbuhan (%)",
        "average_transaction_value": "Rata-rata nilai transaksi"
    })
    st.dataframe(table.style.format({
        "Total penjualan": "{:,.2f}", "Transaksi tercatat": "{:,.0f}",
        "Unit terjual": "{:,.0f}", "Pertumbuhan (%)": "{:,.2f}",
        "Rata-rata nilai transaksi": "{:,.2f}"
    }, na_rep="Belum tersedia", decimal=",", thousands="."),
        use_container_width=True, hide_index=True)

st.divider()
st.header("Perbandingan gerai dan kategori")
left, right = st.columns(2)
with left:
    st.subheader("Penjualan per gerai")
    sales_by_store = data_display.groupby("store_location", as_index=False).agg(
        total_sales=("sales", "sum")
    ).sort_values("total_sales", ascending=False)
    fig_store = px.bar(sales_by_store, x="store_location", y="total_sales",
                       color="store_location", text_auto=".2s",
                       labels={"store_location": "Gerai", "total_sales": "Nilai penjualan"},
                       color_discrete_sequence=["#087F8C", "#7656B5", "#D98B39"])
    fig_store.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_store, use_container_width=True, key="stores")
    st.caption("Hanya gerai yang dipilih muncul pada grafik.")
with right:
    st.subheader("Kontribusi penjualan kategori")
    sales_by_category = data_display.groupby(
        "product_category", as_index=False
    ).agg(total_sales=("sales", "sum")).sort_values(
        "total_sales", ascending=False
    )
    fig_category = px.pie(sales_by_category, names="product_category",
                          values="total_sales", hole=0.55)
    fig_category.update_traces(textinfo="none", hovertemplate=(
        "%{label}<br>Penjualan: %{value:,.2f}<br>"
        "Kontribusi: %{percent}<extra></extra>"
    ))
    fig_category.update_layout(height=400, legend_title_text="Kategori")
    st.plotly_chart(fig_category, use_container_width=True, key="categories")
    st.caption("Kontribusi terhadap total dalam pilihan gerai yang sama.")

st.header("Produk dan pola waktu transaksi")
left, right = st.columns(2)
with left:
    # [Ide 2] Pilihan Top 10 vs Bottom 10 Produk
    st.subheader("Peringkat produk berdasarkan penjualan")
    rank_type = st.radio(
        "Tampilkan:",
        ["10 Produk Tertinggi (Top 10)", "10 Produk Terendah (Bottom 10)"],
        horizontal=True,
        key="rank_type_selector"
    )

    if "Tertinggi" in rank_type:
        products_subset = data_display.groupby(
            ["product_id", "product_detail"], as_index=False
        ).agg(total_sales=("sales", "sum")).nlargest(
            10, "total_sales"
        ).sort_values("total_sales", ascending=True)
        bar_color = ["#7656B5"]
        caption_product = "Peringkat nilai penjualan tertinggi (Top 10), bukan unit atau keuntungan."
    else:
        products_subset = data_display.groupby(
            ["product_id", "product_detail"], as_index=False
        ).agg(total_sales=("sales", "sum")).nsmallest(
            10, "total_sales"
        ).sort_values("total_sales", ascending=True)
        bar_color = ["#D9534F"]
        caption_product = "Peringkat nilai penjualan terendah (Bottom 10) untuk evaluasi menu/dead stock."

    products_subset["label"] = (products_subset["product_detail"] + " ("
                               + products_subset["product_id"].astype(str) + ")")
    fig_products = px.bar(products_subset, x="total_sales", y="label",
                          orientation="h", color_discrete_sequence=bar_color,
                          labels={"total_sales": "Nilai penjualan", "label": "Produk"})
    fig_products.update_layout(height=470, yaxis={
        "categoryorder": "array", "categoryarray": products_subset["label"].tolist()
    })
    st.plotly_chart(fig_products, use_container_width=True, key="products")
    st.caption(caption_product)

with right:
    st.subheader("Transaksi tercatat menurut jam")
    by_hour = data_display.groupby("hour", as_index=False).agg(
        total_transactions=("transaction_id", "nunique")
    ).sort_values("hour")
    by_hour["label"] = by_hour["hour"].astype(int).astype(str).str.zfill(2)+":00"
    fig_hour = px.bar(by_hour, x="label", y="total_transactions",
                      color_discrete_sequence=["#D98B39"],
                      labels={"label": "Jam", "total_transactions": "Transaksi tercatat"})
    fig_hour.update_layout(height=470, xaxis_type="category")
    st.plotly_chart(fig_hour, use_container_width=True, key="hours")
    st.caption("Akumulasi transaksi menurut jam dalam cakupan pilihan.")

# [Ide 1] Matriks Jam Sibuk Operasional (Heatmap Hari vs Jam)
st.subheader("Matriks jam sibuk operasional (Hari vs Jam)")
day_order = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
heat_data = data_display.groupby(["day_name_id", "hour"])["transaction_id"].nunique().reset_index()

if not heat_data.empty:
    pivot_heat = heat_data.pivot(index="day_name_id", columns="hour", values="transaction_id").reindex(day_order).fillna(0)
    fig_heatmap = px.imshow(
        pivot_heat,
        labels=dict(x="Jam Transaksi", y="Hari", color="Transaksi"),
        x=[f"{int(h):02d}:00" for h in pivot_heat.columns],
        y=pivot_heat.index,
        color_continuous_scale="Oranges",
        aspect="auto"
    )
    fig_heatmap.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap_operational")
    st.caption(
        "Intensitas warna menunjukkan jumlah transaksi unik. "
        "Area paling pekat mengindikasikan jam sibuk utama untuk penentuan jadwal shift barista dan jam restock bahan baku."
    )

st.caption(
    "Transaksi tercatat adalah transaction_id unik, bukan pelanggan unik. "
    "Rata-rata nilainya belum dipastikan sebagai seluruh isi struk pelanggan."
)
