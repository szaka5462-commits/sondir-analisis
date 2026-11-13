# -*- coding: utf-8 -*-
"""
SONDIR GEOTEKNIK
Aplikasi pengolahan data sondir dan estimasi awal kapasitas pondasi tiang.
Jalankan dengan: streamlit run app.py
"""

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    PETA_TERSEDIA = True
except Exception:
    PETA_TERSEDIA = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    PDF_TERSEDIA = True
except Exception:
    PDF_TERSEDIA = False


# -------------------- KONFIGURASI --------------------
st.set_page_config(
    page_title="SONDIR GEOTEKNIK",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {background: #f4f7fb;}
[data-testid="stSidebar"] {background: #10243a;}
[data-testid="stSidebar"] * {color: #ffffff;}
.hero {
  background: linear-gradient(120deg,#10243a,#183b5b,#245b85);
  padding: 26px 30px; border-radius: 15px; color: white; margin-bottom: 20px;
}
.hero h1 {color:white; margin:0; font-size:30px;}
.hero p {color:#d8e5f1; margin:7px 0 0 0;}
div.stButton > button {border-radius:8px; font-weight:600;}
</style>
""", unsafe_allow_html=True)

KOLOM = ["Kedalaman (m)", "qc (kg/cm2)", "JHL (kg/cm)"]

DATA_CONTOH = pd.DataFrame({
    "Kedalaman (m)": [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
    "qc (kg/cm2)": [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45],
    "JHL (kg/cm)": [0, 3, 7, 12, 18, 25, 33, 42, 52, 63, 75, 88, 102, 117, 133, 150, 168, 187, 207, 228, 250],
})


def validasi_data(data):
    if data is None:
        return pd.DataFrame(columns=KOLOM)
    df = pd.DataFrame(data).copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Pemetaan nama kolom umum dari file pengguna
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("kedalaman", "depth", "kedalaman (m)"):
            rename[c] = "Kedalaman (m)"
        elif cl in ("qc", "qc (kg/cm2)", "qc (kg/cm²)"):
            rename[c] = "qc (kg/cm2)"
        elif cl in ("jhl", "jhl (kg/cm)", "jhl (kg/cm)"):
            rename[c] = "JHL (kg/cm)"
    df = df.rename(columns=rename)
    for c in KOLOM:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[KOLOM].dropna()
    df = df[(df[KOLOM] >= 0).all(axis=1)]
    df = df.sort_values("Kedalaman (m)").drop_duplicates("Kedalaman (m)", keep="last")
    return df.reset_index(drop=True)


def baca_file(file):
    if file is None:
        return None
    nama = file.name.lower()
    if nama.endswith(".csv"):
        try:
            raw = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            raw = pd.read_csv(file, encoding="latin-1")
    elif nama.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(file)
    else:
        raise ValueError("Format file harus CSV, XLSX, atau XLS.")
    hasil = validasi_data(raw)
    if len(hasil) < 2:
        raise ValueError("File harus memiliki minimal dua baris data lengkap dengan kolom Kedalaman, qc, dan JHL.")
    return hasil


def interpolasi(df, kolom, kedalaman):
    return float(np.interp(kedalaman, df["Kedalaman (m)"], df[kolom]))


def hitung_analisis_dasar(data):
    df = validasi_data(data)
    if len(df) < 2:
        raise ValueError("Masukkan minimal dua baris data sondir.")
    hasil = df.copy()
    hasil["Interval Kedalaman (m)"] = hasil["Kedalaman (m)"].diff()
    hasil["Delta JHL (kg/cm)"] = hasil["JHL (kg/cm)"].diff()
    interval = hasil["Interval Kedalaman (m)"].replace(0, np.nan)
    hasil["Hambatan Lekat Interval (kg/cm2)"] = hasil["Delta JHL (kg/cm)"] / interval
    ringkasan = {
        "Jumlah data": len(df),
        "Kedalaman maksimum (m)": float(df["Kedalaman (m)"].max()),
        "qc minimum (kg/cm²)": float(df["qc (kg/cm2)"].min()),
        "qc maksimum (kg/cm²)": float(df["qc (kg/cm2)"].max()),
        "qc rata-rata (kg/cm²)": float(df["qc (kg/cm2)"].mean()),
        "JHL maksimum (kg/cm)": float(df["JHL (kg/cm)"].max()),
    }
    return hasil, ringkasan


def hitung_pondasi(data, diameter_m, kedalaman_m, fk):
    """
    Estimasi awal mengikuti pendekatan sederhana pada draft pengguna:
    qb = min(0.4 * qc rata-rata zona ujung, 40 kg/cm2)
    Qp = qb * Ap; Qs = JHL pada ujung * keliling; Qu = Qp + Qs.
    Hasil wajib diverifikasi dengan metode desain dan standar yang berlaku.
    """
    df = validasi_data(data)
    if len(df) < 2:
        raise ValueError("Data sondir belum mencukupi.")
    if diameter_m <= 0 or kedalaman_m <= 0 or fk <= 0:
        raise ValueError("Diameter, kedalaman, dan faktor keamanan harus lebih dari nol.")
    zmax = float(df["Kedalaman (m)"].max())
    if kedalaman_m > zmax:
        raise ValueError(f"Kedalaman tiang ({kedalaman_m:g} m) melebihi kedalaman data ({zmax:g} m).")

    D_cm = diameter_m * 100.0
    Ap_cm2 = math.pi * D_cm**2 / 4.0
    keliling_cm = math.pi * D_cm

    z_atas = max(0.0, kedalaman_m - 8.0 * diameter_m)
    z_bawah = min(zmax, kedalaman_m + 4.0 * diameter_m)
    zona = df[(df["Kedalaman (m)"] >= z_atas) & (df["Kedalaman (m)"] <= z_bawah)]
    if len(zona):
        qc_avg = float(zona["qc (kg/cm2)"].mean())
    else:
        qc_avg = interpolasi(df, "qc (kg/cm2)", kedalaman_m)

    qb = min(0.4 * qc_avg, 40.0)
    Qp_kgf = qb * Ap_cm2
    JHL_tip = interpolasi(df, "JHL (kg/cm)", kedalaman_m)
    Qs_kgf = JHL_tip * keliling_cm
    Qu_kgf = Qp_kgf + Qs_kgf
    Qizin_kgf = Qu_kgf / fk
    kgf_to_kN = 0.00980665

    return {
        "Diameter tiang (m)": diameter_m,
        "Kedalaman tiang (m)": kedalaman_m,
        "Faktor keamanan": fk,
        "Luas ujung Ap (cm²)": Ap_cm2,
        "Keliling tiang (cm)": keliling_cm,
        "qc rata-rata zona ujung (kg/cm²)": qc_avg,
        "qb (kg/cm²)": qb,
        "JHL pada ujung (kg/cm)": JHL_tip,
        "Qp (kgf)": Qp_kgf,
        "Qs (kgf)": Qs_kgf,
        "Qu (kgf)": Qu_kgf,
        "Q izin (kgf)": Qizin_kgf,
        "Qp (kN)": Qp_kgf * kgf_to_kN,
        "Qs (kN)": Qs_kgf * kgf_to_kN,
        "Qu (kN)": Qu_kgf * kgf_to_kN,
        "Q izin (kN)": Qizin_kgf * kgf_to_kN,
        "Zona qc atas (m)": z_atas,
        "Zona qc bawah (m)": z_bawah,
    }


def validasi_data_qc3d(data):
    kolom = ["ID Titik", "Latitude", "Longitude", "Kedalaman (m)", "qc (kg/cm2)"]
    df3 = pd.DataFrame(data).copy()
    for c in kolom:
        if c not in df3.columns:
            df3[c] = np.nan
    df3 = df3[kolom]
    for c in ["Latitude", "Longitude", "Kedalaman (m)", "qc (kg/cm2)"]:
        df3[c] = pd.to_numeric(df3[c], errors="coerce")
    df3["ID Titik"] = df3["ID Titik"].fillna("").astype(str).str.strip()
    df3 = df3.dropna(subset=["Latitude", "Longitude", "Kedalaman (m)", "qc (kg/cm2)"])
    df3 = df3[
        df3["ID Titik"].ne("") &
        df3["Latitude"].between(-90, 90) &
        df3["Longitude"].between(-180, 180) &
        (df3["Kedalaman (m)"] >= 0) &
        (df3["qc (kg/cm2)"] >= 0)
    ]
    return df3.reset_index(drop=True)


def buat_gambar_grafik(df, data3d=None):
    gambar = []
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["qc (kg/cm2)"], df["Kedalaman (m)"], marker="o")
    ax.invert_yaxis()
    ax.set_xlabel("qc (kg/cm²)")
    ax.set_ylabel("Kedalaman (m)")
    ax.set_title("Profil qc terhadap Kedalaman")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    b.seek(0)
    gambar.append(("Grafik qc", b))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["JHL (kg/cm)"], df["Kedalaman (m)"], marker="o")
    ax.invert_yaxis()
    ax.set_xlabel("JHL (kg/cm)")
    ax.set_ylabel("Kedalaman (m)")
    ax.set_title("Profil JHL terhadap Kedalaman")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    b.seek(0)
    gambar.append(("Grafik JHL", b))

    if data3d is not None and len(data3d) >= 1:
        fig = plt.figure(figsize=(7.2, 5.0))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            data3d["Longitude"], data3d["Latitude"], data3d["Kedalaman (m)"],
            c=data3d["qc (kg/cm2)"], cmap="viridis", s=35
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Kedalaman (m)")
        ax.set_zlim(float(data3d["Kedalaman (m)"].max()), 0)
        ax.set_title("Persebaran Nilai qc 3D")
        fig.colorbar(sc, ax=ax, shrink=0.65, label="qc (kg/cm²)")
        fig.tight_layout()
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        b.seek(0)
        gambar.append(("Peta Persebaran qc 3D", b))
    return gambar


def buat_pdf(df, hasil_pile=None, data3d=None):
    if not PDF_TERSEDIA:
        raise RuntimeError("Library reportlab belum terpasang. Periksa requirements.txt.")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("LAPORAN DATA SONDIR - SONDIR GEOTEKNIK", styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Tanggal dibuat: {datetime.now().strftime('%d-%m-%Y %H:%M')}", styles["Normal"]),
        Spacer(1, 10),
        Paragraph("1. Tabel Data Sondir", styles["Heading2"]),
    ]
    tabel = [KOLOM] + df[KOLOM].round(3).astype(str).values.tolist()
    t = Table(tabel, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173653")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .4, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef3f8")]),
    ]))
    story.extend([t, Spacer(1, 12)])

    if hasil_pile:
        story.append(Paragraph("2. Estimasi Kapasitas Pondasi Tiang", styles["Heading2"]))
        data = [["Parameter", "Nilai"]] + [
            [k, f"{v:,.3f}" if isinstance(v, (int, float, np.number)) else str(v)]
            for k, v in hasil_pile.items()
        ]
        tt = Table(data, colWidths=[9*cm, 7*cm], repeatRows=1)
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173653")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), .4, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))
        story.extend([tt, Spacer(1, 10)])
        story.append(Paragraph(
            "Catatan: hasil ini merupakan estimasi awal berdasarkan rumus pendekatan sederhana "
            "pada aplikasi, bukan pengganti verifikasi perencana geoteknik terhadap metode dan "
            "standar desain yang berlaku.", styles["Italic"]
        ))

    story.append(Paragraph("3. Grafik Visualisasi", styles["Heading2"]))
    for judul, img_buffer in buat_gambar_grafik(df, data3d):
        story.append(Paragraph(judul, styles["Heading3"]))
        story.append(Image(img_buffer, width=16*cm, height=9*cm))
        story.append(Spacer(1, 8))

    if data3d is not None and len(data3d):
        story.append(Paragraph("4. Data Spasial Persebaran qc", styles["Heading2"]))
        rows = [["ID Titik", "Latitude", "Longitude", "Kedalaman (m)", "qc (kg/cm²)"]]
        rows += data3d.round(3).astype(str).values.tolist()
        tab3 = Table(rows, repeatRows=1)
        tab3.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#173653")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), .4, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 7),
        ]))
        story.append(tab3)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -------------------- SESSION STATE --------------------
if "data_sondir" not in st.session_state:
    st.session_state.data_sondir = DATA_CONTOH.copy()
if "data_titik" not in st.session_state:
    st.session_state.data_titik = pd.DataFrame(columns=["ID Titik", "Latitude", "Longitude", "Kedalaman Maks (m)", "qc Maks (kg/cm2)", "JHL Maks (kg/cm)"])
if "hasil_pile" not in st.session_state:
    st.session_state.hasil_pile = None
if "data_qc_3d" not in st.session_state:
    st.session_state.data_qc_3d = pd.DataFrame(columns=[
        "ID Titik", "Latitude", "Longitude", "Kedalaman (m)", "qc (kg/cm2)"
    ])
if "data_qc_3d" not in st.session_state:
    st.session_state.data_qc_3d = pd.DataFrame(columns=[
        "ID Titik", "Latitude", "Longitude", "Kedalaman (m)", "qc (kg/cm2)"
    ])

# -------------------- HEADER & NAVIGASI --------------------
st.markdown("""
<div class="hero">
  <h1>🏗️ SONDIR GEOTEKNIK</h1>
  <p>GEOTECHNICAL ENGINEERING DATA MANAGEMENT SYSTEM</p>
  <p>Pengolahan data sondir, grafik qc dan JHL, estimasi awal pondasi tiang, serta laporan.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🏗️ SONDIR GEOTEKNIK")
st.sidebar.caption("Aplikasi akademik pengolahan data sondir")
menu = st.sidebar.radio("MENU UTAMA", [
    "Dashboard", "Input Data Sondir", "Analisis Data",
    "Perhitungan Pondasi", "Grafik Visualisasi",
    "Peta Lokasi", "Peta Persebaran qc 3D", "Multi Titik Sondir", "Laporan PDF"
])
st.sidebar.divider()
st.sidebar.info("Data contoh tersedia saat aplikasi pertama kali dibuka. Ganti dengan data pengujian sebenarnya melalui menu Input Data Sondir.")

df = validasi_data(st.session_state.data_sondir)

# -------------------- DASHBOARD --------------------
if menu == "Dashboard":
    st.title("Dashboard")
    st.write("Ringkasan kondisi data sondir yang sedang aktif.")
    a, b, c, d = st.columns(4)
    a.metric("Jumlah data", f"{len(df)} baris")
    b.metric("Kedalaman maksimum", f"{df['Kedalaman (m)'].max():.2f} m" if len(df) else "-")
    c.metric("qc maksimum", f"{df['qc (kg/cm2)'].max():.2f} kg/cm²" if len(df) else "-")
    d.metric("JHL maksimum", f"{df['JHL (kg/cm)'].max():.2f} kg/cm" if len(df) else "-")
    st.subheader("Pratinjau data sondir")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if len(df) >= 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["qc (kg/cm2)"], y=df["Kedalaman (m)"], mode="lines+markers", name="qc (kg/cm²)"))
        fig.update_yaxes(autorange="reversed", title="Kedalaman (m)")
        fig.update_xaxes(title="qc (kg/cm²)")
        fig.update_layout(height=420, title="Profil qc terhadap Kedalaman")
        st.plotly_chart(fig, use_container_width=True)

# -------------------- INPUT DATA --------------------
elif menu == "Input Data Sondir":
    st.title("Input dan Edit Data Sondir")
    st.write("Unggah file CSV/Excel atau edit tabel secara langsung. Nama kolom yang dikenali: Kedalaman, qc, dan JHL.")
    file = st.file_uploader("Pilih file data sondir", type=["csv", "xlsx", "xls"], key="upload_sondir")
    if file is not None:
        try:
            uploaded_df = baca_file(file)
            st.session_state.data_sondir = uploaded_df
            st.success(f"Berhasil membaca {len(uploaded_df)} baris data.")
            df = uploaded_df
        except Exception as e:
            st.error(str(e))
    st.caption("Satuan yang digunakan aplikasi: kedalaman dalam m, qc dalam kg/cm², JHL dalam kg/cm.")
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "Kedalaman (m)": st.column_config.NumberColumn("Kedalaman (m)", min_value=0.0, format="%.3f"),
            "qc (kg/cm2)": st.column_config.NumberColumn("qc (kg/cm²)", min_value=0.0, format="%.3f"),
            "JHL (kg/cm)": st.column_config.NumberColumn("JHL (kg/cm)", min_value=0.0, format="%.3f"),
        },
        key="editor_data_sondir"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Simpan Perubahan", type="primary"):
            checked = validasi_data(edited)
            if len(checked) < 2:
                st.error("Data harus memiliki minimal dua baris lengkap.")
            else:
                st.session_state.data_sondir = checked
                st.success("Perubahan data berhasil disimpan.")
                st.rerun()
    with col2:
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Unduh Data CSV", csv, "data_sondir.csv", "text/csv")

# -------------------- ANALISIS DATA --------------------
elif menu == "Analisis Data":
    st.title("Analisis Data Sondir")
    if len(df) < 2:
        st.warning("Masukkan minimal dua baris data terlebih dahulu.")
    else:
        hasil, ringkasan = hitung_analisis_dasar(df)
        cols = st.columns(3)
        for i, (k, v) in enumerate(ringkasan.items()):
            cols[i % 3].metric(k, f"{v:,.2f}" if isinstance(v, float) else v)
        st.subheader("Tabel hasil pengolahan")
        st.dataframe(hasil.round(3), use_container_width=True, hide_index=True)
        st.download_button("Unduh hasil analisis CSV", hasil.to_csv(index=False).encode("utf-8-sig"), "hasil_analisis_sondir.csv", "text/csv")
        st.info("Hambatan lekat interval dihitung dari perubahan JHL dibagi interval kedalaman. Periksa kesesuaian satuan dan format data lapangan sebelum menggunakan hasil.")

# -------------------- PONDASI --------------------
elif menu == "Perhitungan Pondasi":
    st.title("Estimasi Awal Kapasitas Pondasi Tiang")
    if len(df) < 2:
        st.warning("Masukkan data sondir terlebih dahulu.")
    else:
        zmax = float(df["Kedalaman (m)"].max())
        c1, c2, c3 = st.columns(3)
        with c1:
            diameter = st.number_input("Diameter tiang (m)", min_value=0.05, max_value=5.0, value=0.50, step=0.05)
        with c2:
            kedalaman = st.number_input("Kedalaman ujung tiang (m)", min_value=0.10, max_value=max(0.10, zmax), value=min(10.0, zmax), step=0.5)
        with c3:
            fk = st.number_input("Faktor keamanan", min_value=1.0, max_value=10.0, value=2.5, step=0.1)
        if st.button("Hitung Kapasitas Pondasi", type="primary"):
            try:
                st.session_state.hasil_pile = hitung_pondasi(df, diameter, kedalaman, fk)
            except Exception as e:
                st.error(str(e))
        if st.session_state.hasil_pile:
            hp = st.session_state.hasil_pile
            st.subheader("Hasil perhitungan")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Qp", f"{hp['Qp (kN)']:,.2f} kN")
            x2.metric("Qs", f"{hp['Qs (kN)']:,.2f} kN")
            x3.metric("Qu", f"{hp['Qu (kN)']:,.2f} kN")
            x4.metric("Q izin", f"{hp['Q izin (kN)']:,.2f} kN")
            st.dataframe(pd.DataFrame({"Parameter": list(hp.keys()), "Nilai": list(hp.values())}), use_container_width=True, hide_index=True)
            st.warning("Catatan teknis: rumus yang dipakai adalah pendekatan empiris sederhana dari draft awal (qb = min(0,4 qc rata-rata zona ujung; 40 kg/cm²)). Ini belum boleh dianggap sebagai verifikasi final metode Meyerhof atau desain pondasi. Cocokkan dengan referensi/metode yang diwajibkan dosen, kondisi tanah, jenis tiang, dan standar yang berlaku.")
            st.download_button("Unduh hasil pondasi CSV", pd.DataFrame([hp]).to_csv(index=False).encode("utf-8-sig"), "hasil_pondasi.csv", "text/csv")

# -------------------- GRAFIK --------------------
elif menu == "Grafik Visualisasi":
    st.title("Grafik Visualisasi Data Sondir")
    if len(df) < 2:
        st.warning("Data belum mencukupi untuk membuat grafik.")
    else:
        opsi = st.selectbox("Pilih grafik", ["Profil qc terhadap kedalaman", "Profil JHL terhadap kedalaman", "Grafik qc dan JHL terpisah"])
        if opsi == "Profil qc terhadap kedalaman":
            fig = px.line(df, x="qc (kg/cm2)", y="Kedalaman (m)", markers=True, title="Profil qc terhadap Kedalaman")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        elif opsi == "Profil JHL terhadap kedalaman":
            fig = px.line(df, x="JHL (kg/cm)", y="Kedalaman (m)", markers=True, title="Profil JHL terhadap Kedalaman")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            left, right = st.columns(2)
            f1 = px.line(df, x="qc (kg/cm2)", y="Kedalaman (m)", markers=True, title="qc vs Kedalaman")
            f1.update_yaxes(autorange="reversed")
            f2 = px.line(df, x="JHL (kg/cm)", y="Kedalaman (m)", markers=True, title="JHL vs Kedalaman")
            f2.update_yaxes(autorange="reversed")
            left.plotly_chart(f1, use_container_width=True)
            right.plotly_chart(f2, use_container_width=True)

# -------------------- PETA --------------------
elif menu == "Peta Lokasi":
    st.title("Peta Lokasi Titik Sondir")
    if not PETA_TERSEDIA:
        st.error("Komponen peta belum tersedia. Pastikan folium dan streamlit-folium tercantum di requirements.txt.")
    else:
        st.write("Masukkan koordinat lokasi titik sondir untuk menampilkannya pada peta.")
        c1, c2, c3 = st.columns(3)
        with c1:
            titik_id = st.text_input("ID titik", value="S-01")
        with c2:
            lat = st.number_input("Latitude", value=-7.4700, format="%.6f")
        with c3:
            lon = st.number_input("Longitude", value=110.2200, format="%.6f")
        if st.button("Simpan Titik Lokasi"):
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                st.error("Koordinat latitude/longitude tidak valid.")
            else:
                row = {"ID Titik": titik_id, "Latitude": lat, "Longitude": lon,
                       "Kedalaman Maks (m)": float(df["Kedalaman (m)"].max()) if len(df) else np.nan,
                       "qc Maks (kg/cm2)": float(df["qc (kg/cm2)"].max()) if len(df) else np.nan,
                       "JHL Maks (kg/cm)": float(df["JHL (kg/cm)"].max()) if len(df) else np.nan}
                st.session_state.data_titik = pd.concat([st.session_state.data_titik, pd.DataFrame([row])], ignore_index=True)
                st.success("Titik lokasi disimpan.")
        titik = st.session_state.data_titik
        if len(titik):
            center = [titik["Latitude"].mean(), titik["Longitude"].mean()]
            m = folium.Map(location=center, zoom_start=14)
            for _, r in titik.iterrows():
                folium.Marker([r["Latitude"], r["Longitude"]], tooltip=str(r["ID Titik"]),
                              popup=f"{r['ID Titik']}<br>Kedalaman: {r['Kedalaman Maks (m)']} m").add_to(m)
            st_folium(m, use_container_width=True, height=480)
            st.dataframe(titik, use_container_width=True, hide_index=True)
            st.download_button("Unduh daftar titik CSV", titik.to_csv(index=False).encode("utf-8-sig"), "titik_sondir.csv", "text/csv")
        else:
            st.info("Belum ada titik yang disimpan.")

# -------------------- PETA QC 3D --------------------
elif menu == "Peta Persebaran qc 3D":
    st.title("Peta Persebaran Nilai qc 3D")
    st.write("Visualisasi 3D menggabungkan koordinat lokasi, kedalaman, dan nilai qc. Setiap baris mewakili satu pengukuran pada satu titik sondir dan satu kedalaman.")
    st.info("Format kolom: ID Titik, Latitude, Longitude, Kedalaman (m), qc (kg/cm2). Masukkan beberapa lokasi dan kedalaman agar pola persebarannya lebih informatif.")
    template = pd.DataFrame({
        "ID Titik": ["S-01", "S-01", "S-02", "S-02"],
        "Latitude": [-7.4700, -7.4700, -7.4710, -7.4710],
        "Longitude": [110.2200, 110.2200, 110.2210, 110.2210],
        "Kedalaman (m)": [1.0, 2.0, 1.0, 2.0],
        "qc (kg/cm2)": [10.0, 18.0, 12.0, 22.0],
    })
    st.download_button("Unduh template CSV qc 3D", template.to_csv(index=False).encode("utf-8-sig"), "template_qc_3d.csv", "text/csv")
    file3d = st.file_uploader("Unggah data spasial CSV/Excel", type=["csv", "xlsx", "xls"], key="upload_qc3d")
    if file3d is not None:
        try:
            raw3d = pd.read_csv(file3d) if file3d.name.lower().endswith(".csv") else pd.read_excel(file3d)
            st.session_state.data_qc_3d = validasi_data_qc3d(raw3d)
            st.success(f"Data spasial berhasil dimuat: {len(st.session_state.data_qc_3d)} baris valid.")
        except Exception as e:
            st.error(f"Gagal membaca data spasial: {e}")

    st.subheader("Tabel data spasial (dapat diedit)")
    edited3d = st.data_editor(
        st.session_state.data_qc_3d if len(st.session_state.data_qc_3d) else template,
        num_rows="dynamic", use_container_width=True, hide_index=True, key="editor_qc3d"
    )
    if st.button("Simpan Data qc 3D", type="primary"):
        clean3d = validasi_data_qc3d(edited3d)
        if clean3d.empty:
            st.error("Belum ada baris valid. Periksa ID titik, koordinat, kedalaman, dan qc.")
        else:
            st.session_state.data_qc_3d = clean3d
            st.success(f"{len(clean3d)} baris data qc 3D tersimpan.")
            st.rerun()

    data3d = validasi_data_qc3d(st.session_state.data_qc_3d)
    if len(data3d):
        fig3 = px.scatter_3d(
            data3d, x="Longitude", y="Latitude", z="Kedalaman (m)",
            color="qc (kg/cm2)", hover_name="ID Titik",
            color_continuous_scale="Viridis",
            title="Peta Persebaran Nilai qc 3D"
        )
        fig3.update_scenes(zaxis_autorange="reversed")
        fig3.update_layout(height=650)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Warna menunjukkan besar nilai qc. Sumbu vertikal menunjukkan kedalaman dan dibalik agar kedalaman bertambah ke bawah.")
        st.download_button("Unduh data qc 3D CSV", data3d.to_csv(index=False).encode("utf-8-sig"), "data_qc_3d.csv", "text/csv")
    else:
        st.warning("Belum ada data spasial tersimpan. Unggah atau isi tabel, lalu klik Simpan Data qc 3D.")

# -------------------- MULTI TITIK --------------------
elif menu == "Multi Titik Sondir":
    st.title("Pengelolaan Multi Titik Sondir")
    st.write("Simpan ringkasan beberapa titik pengujian. Untuk setiap titik, masukkan data pada menu Input Data Sondir, lalu simpan ringkasannya di sini.")
    nama = st.text_input("Nama/ID titik sondir", value=f"S-{len(st.session_state.data_titik)+1:02d}")
    if st.button("Tambahkan ringkasan data aktif"):
        if len(df) < 2:
            st.error("Data aktif belum mencukupi.")
        else:
            row = {"ID Titik": nama, "Latitude": np.nan, "Longitude": np.nan,
                   "Kedalaman Maks (m)": float(df["Kedalaman (m)"].max()),
                   "qc Maks (kg/cm2)": float(df["qc (kg/cm2)"].max()),
                   "JHL Maks (kg/cm)": float(df["JHL (kg/cm)"].max())}
            st.session_state.data_titik = pd.concat([st.session_state.data_titik, pd.DataFrame([row])], ignore_index=True)
            st.success("Ringkasan titik ditambahkan.")
    st.subheader("Daftar titik tersimpan")
    st.dataframe(st.session_state.data_titik, use_container_width=True, hide_index=True)
    if len(st.session_state.data_titik):
        st.download_button("Unduh rekap multi titik CSV", st.session_state.data_titik.to_csv(index=False).encode("utf-8-sig"), "rekap_multi_titik.csv", "text/csv")
        if st.button("Hapus seluruh daftar titik"):
            st.session_state.data_titik = st.session_state.data_titik.iloc[0:0].copy()
            st.rerun()

# -------------------- LAPORAN PDF --------------------
elif menu == "Laporan PDF":
    st.title("Laporan PDF")
    st.write("Buat laporan berisi tabel data, hasil estimasi pondasi (jika tersedia), grafik qc, grafik JHL, dan grafik persebaran qc 3D (jika data spasial sudah diisi).")
    st.write(f"Jumlah data aktif: **{len(df)} baris**")
    if not PDF_TERSEDIA:
        st.error("Library reportlab belum tersedia. Periksa requirements.txt dan lakukan reboot aplikasi.")
    elif len(df) < 2:
        st.warning("Masukkan minimal dua baris data sebelum membuat laporan.")
    else:
        try:
            pdf_bytes = buat_pdf(df, st.session_state.hasil_pile, validasi_data_qc3d(st.session_state.data_qc_3d))
            st.download_button("📄 Unduh Laporan PDF", pdf_bytes, f"laporan_sondir_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf", type="primary")
        except Exception as e:
            st.error(f"Gagal membuat PDF: {e}")

st.sidebar.caption("Versi akademik | Periksa kembali data dan asumsi perhitungan sebelum digunakan untuk desain.")
