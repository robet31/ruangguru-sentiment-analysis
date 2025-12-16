# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import re
import ast
from collections import Counter
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Konfigurasi Streamlit
st.set_page_config(
    page_title="📊 Analisis Sentimen Ruangguru",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
    }
    .sentiment-positive { color: #27ae60; font-weight: bold; }
    .sentiment-neutral { color: #f39c12; font-weight: bold; }
    .sentiment-negative { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Setup Sastrawi
@st.cache_resource
def setup_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = setup_stemmer()

# Setup komponen lain yang diperlukan
@st.cache_data
def load_slang_dict():
    try:
        slang_df = pd.read_csv('Ruangguru/slang_indo.csv', header=None, names=['alay', 'baku'])
        return dict(zip(slang_df['alay'], slang_df['baku']))
    except:
        st.warning("⚠️ Kamus slang tidak ditemukan, menggunakan kamus kosong")
        return {}

slang_dict = load_slang_dict()

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    """Memuat data preprocessing dari file Excel"""
    try:
        df = pd.read_excel('Ruangguru/data_preprocessing_ruangguru.xlsx')
        # Konversi string list ke list asli
        for col in ['tokenisasi', 'normalisasi', 'stopwords']:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None

# Fungsi preprocessing teks
def preprocess_text(text):
    """Preprocessing teks untuk prediksi"""
    # Case folding
    text = text.lower()
    
    # Cleaning
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenisasi
    tokens = text.split()
    
    # Normalisasi slang
    tokens = [slang_dict.get(word, word) for word in tokens]
    
    # Stopword removal
    stop_words = set(nltk.corpus.stopwords.words('indonesian'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    # Stemming
    tokens = [stemmer.stem(w) for w in tokens]
    
    return ' '.join(tokens)

# Fungsi untuk membuat word cloud
def create_wordcloud(text_data, title):
    """Membuat word cloud dari teks"""
    text = ' '.join(text_data.dropna().astype(str))
    if text.strip():
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            font_path=None
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        return fig
    return None

# Fungsi untuk visualisasi distribusi sentimen
def plot_sentiment_distribution(data, title, colors=['#e74c3c', '#f39c12', '#27ae60']):
    """Membuat visualisasi distribusi sentimen"""
    sentiment_counts = data['Label'].value_counts()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        marker_color=colors,
        text=sentiment_counts.values,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Jumlah: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Sentimen",
        yaxis_title="Jumlah Data",
        template='plotly_white',
        height=400,
        font=dict(size=14)
    )
    return fig

# Fungsi untuk confusion matrix
def plot_confusion_matrix(cm, labels):
    """Membuat confusion matrix dengan Plotly"""
    max_value = np.max(cm)
    
    fig = px.imshow(
        cm,
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title="Confusion Matrix - Voting Classifier",
        height=500,
        font=dict(size=14)
    )
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(size=20, color='white' if cm[i][j] > max_value / 2 else 'black')
            )
    
    return fig

# Fungsi untuk performance metrics
def plot_performance_comparison(scores):
    """Membuat visualisasi perbandingan performa model"""
    models = list(scores.keys())
    f1_scores = list(scores.values())
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=f1_scores,
        marker_color=['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c'],
        text=[f"{score:.2%}" for score in f1_scores],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>F1 Score: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Perbandingan Performa Model (Macro F1 Score)",
        xaxis_title="Model Algoritma",
        yaxis_title="Macro F1 Score",
        template='plotly_white',
        height=500,
        font=dict(size=14),
        yaxis=dict(range=[0, 1.1])
    )
    
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">📊 Dashboard Analisis Sentimen Ruangguru</h1>', unsafe_allow_html=True)
    
    st.sidebar.title("🎛️ Kontrol Panel")
    st.sidebar.info("Aplikasi ini menganalisis sentimen pengguna terhadap aplikasi Ruangguru")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Mohon pastikan file 'data_preprocessing_ruangguru.xlsx' ada di folder Ruangguru!")
        return
    
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Total Data</h3>
                <h2>{}</h2>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>Sentimen Positif</h3>
                <h2 class='sentiment-positive'>{}</h2>
            </div>
        """.format(len(df[df['Label'] == 'Positif'])), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>Sentimen Netral</h3>
                <h2 class='sentiment-neutral'>{}</h2>
            </div>
        """.format(len(df[df['Label'] == 'Netral'])), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <h3>Sentimen Negatif</h3>
                <h2 class='sentiment-negative'>{}</h2>
            </div>
        """.format(len(df[df['Label'] == 'Negatif'])), unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distribusi Sentimen",
        "🤖 Performa Model",
        "💬 Contoh Teks",
        "☁️ Word Cloud",
        "🔍 Prediksi Sentimen"  # TAB BARU
    ])
    
    # Tab 1: Distribusi Sentimen
    with tab1:
        st.markdown('<h2 class="sub-header">Distribusi Sentimen</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_sentiment_distribution(df, "Distribusi Sentimen Original"),
                use_container_width=True
            )
        
        st.markdown("### Sebelum vs Sesudah SMOTE")
        
        smote_data = {
            'Model': ['Original', 'Setelah SMOTE', 'Original', 'Setelah SMOTE', 'Original', 'Setelah SMOTE'],
            'Sentimen': ['Positif', 'Positif', 'Netral', 'Netral', 'Negatif', 'Negatif'],
            'Jumlah': [1312, 5245, 1027, 5245, 463, 5245]
        }
        
        fig_smote = px.bar(
            smote_data,
            x='Sentimen',
            y='Jumlah',
            color='Model',
            barmode='group',
            color_discrete_sequence=['#95a5a6', '#3498db'],
            text='Jumlah',
            title="Perbandingan Distribusi Data (Sebelum vs Sesudah SMOTE)"
        )
        
        fig_smote.update_layout(
            height=500,
            font=dict(size=14),
            template='plotly_white'
        )
        
        st.plotly_chart(fig_smote, use_container_width=True)
    
    # Tab 2: Performa Model
    with tab2:
        st.markdown('<h2 class="sub-header">Performa Model Machine Learning</h2>', unsafe_allow_html=True)
        
        scores = {
            'KNN': 0.42,
            'NaiveBayes': 0.61,
            'SVM': 0.64,
            'Hard Voting': 0.603,
            'Soft Voting': 0.636
        }
        
        st.plotly_chart(
            plot_performance_comparison(scores),
            use_container_width=True
        )
        
        st.markdown('<h3 class="sub-header">Confusion Matrix - Soft Voting</h3>', unsafe_allow_html=True)
        
        cm = [[326, 107, 30],
              [251, 581, 195],
              [ 94, 278, 940]]
        labels = ['Negatif', 'Netral', 'Positif']
        
        st.plotly_chart(
            plot_confusion_matrix(cm, labels),
            use_container_width=True
        )
        
        st.markdown('<h3 class="sub-header">Classification Report Detail</h3>', unsafe_allow_html=True)
        
        report_data = {
            'Label': ['Negatif', 'Netral', 'Positif', 'Akurasi', 'Macro Avg', 'Weighted Avg'],
            'Precision': ['37.00%', '68.00%', '89.00%', '', '65.00%', '73.00%'],
            'Recall': ['70.00%', '57.00%', '72.00%', '66.00%', '66.00%', '66.00%'],
            'F1-Score': ['49.00%', '62.00%', '80.00%', '66.00%', '64.00%', '68.00%']
        }
        
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)
    
    # Tab 3: Contoh Teks
    with tab3:
        st.markdown('<h2 class="sub-header">Contoh Teks per Kategori Sentimen</h2>', unsafe_allow_html=True)
        
        for sentiment in ['Positif', 'Netral', 'Negatif']:
            with st.expander(f"📌 Contoh Teks {sentiment}"):
                sample_texts = df[df['Label'] == sentiment]['text'].head(5)
                for i, text in enumerate(sample_texts, 1):
                    st.markdown(f"**{i}.** {text}")
    
    # Tab 4: Word Cloud
    with tab4:
        st.markdown('<h2 class="sub-header">Word Cloud per Kategori Sentimen</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ☁️ Word Cloud (Setelah Preprocessing)")
            
            fig_all = create_wordcloud(df['stemming'], "Seluruh Data")
            if fig_all:
                st.pyplot(fig_all, use_container_width=True)
            
            for sentiment in ['Positif', 'Netral', 'Negatif']:
                st.markdown(f"**Sentimen: <span class='sentiment-{sentiment.lower()}'>{sentiment}</span>**", unsafe_allow_html=True)
                data_sentiment = df[df['Label'] == sentiment]
                fig = create_wordcloud(data_sentiment['stemming'], f"Word Cloud - {sentiment}")
                if fig:
                    st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ☁️ Word Cloud (Teks Asli)")
            
            fig_all_orig = create_wordcloud(df['text'], "Seluruh Data (Original)")
            if fig_all_orig:
                st.pyplot(fig_all_orig, use_container_width=True)
            
            for sentiment in ['Positif', 'Netral', 'Negatif']:
                st.markdown(f"**Sentimen: <span class='sentiment-{sentiment.lower()}'>{sentiment}</span>**", unsafe_allow_html=True)
                data_sentiment = df[df['Label'] == sentiment]
                fig = create_wordcloud(data_sentiment['text'], f"Word Cloud - {sentiment} (Original)")
                if fig:
                    st.pyplot(fig, use_container_width=True)
    
    # Tab 5: Prediksi Sentimen
    with tab5:
        st.markdown('<h2 class="sub-header">🔍 Prediksi Sentimen untuk Teks Baru</h2>', unsafe_allow_html=True)
        
        user_text = st.text_area(
            "📝 Masukkan komentar atau review tentang Ruangguru:",
            placeholder="Contoh: aplikasi ini sangat membantu saya belajar matematika",
            height=150
        )
        
        if st.button("🔮 Prediksi Sentimen", type="primary"):
            if user_text.strip() == "":
                st.warning("⚠️ Mohon masukkan teks terlebih dahulu!")
            else:
                with st.spinner('🔄 Sedang menganalisis sentimen...'):
                    try:
                        # Preprocessing
                        st.markdown("#### Langkah Preprocessing:")
                        
                        # Case folding
                        case_folded = user_text.lower()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**1. Case Folding**")
                            st.code(case_folded, language='text')
                        
                        # Cleaning
                        cleaned = re.sub(r'http\S+|www\S+', '', case_folded)
                        cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
                        cleaned = re.sub(r'[^a-z\s]', '', cleaned)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        with col2:
                            st.markdown("**2. Cleaning**")
                            st.code(cleaned, language='text')
                        
                        # Tokenisasi
                        tokens = cleaned.split()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**3. Tokenisasi**")
                            st.code(str(tokens), language='text')
                        
                        # Normalisasi slang
                        normalized = [slang_dict.get(word, word) for word in tokens if word in slang_dict]
                        with col2:
                            st.markdown("**4. Normalisasi Slang**")
                            if normalized:
                                st.code(f"Kata yang dinormalisasi: {normalized}", language='text')
                            else:
                                st.code("Tidak ada kata slang yang ditemukan", language='text')
                        
                        # Stopword removal
                        stop_words = set(nltk.corpus.stopwords.words('indonesian'))
                        filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
                        with col1:
                            st.markdown("**5. Stopword Removal**")
                            st.code(str(filtered_tokens), language='text')
                        
                        # Stemming
                        stemmed = [stemmer.stem(w) for w in filtered_tokens]
                        with col2:
                            st.markdown("**6. Stemming**")
                            st.code(str(stemmed), language='text')
                        
                        final_text = ' '.join(stemmed)
                        
                        # Prediksi sederhana untuk demo
                        positive_words = ['bagus', 'berguna', 'membantu', 'paham', 'menarik', 'terbaik', 'puas',
                                        'mudah', 'jelas', 'nyaman', 'baik', 'suka', 'senang']
                        negative_words = ['jelek', 'buruk', 'sulit', 'boring', 'tidak', 'gak', 'ga', 'masalah',
                                        'lambat', 'error', 'bug', 'crash', 'rusak']
                        
                        positive_count = sum(1 for word in stemmed if word in positive_words)
                        negative_count = sum(1 for word in stemmed if word in negative_words)
                        
                        if positive_count > negative_count:
                            predicted_sentiment = "Positif"
                            confidence = min(positive_count * 0.3 + 0.5, 0.95)
                            color = "#27ae60"
                            emoji = "😊"
                        elif negative_count > positive_count:
                            predicted_sentiment = "Negatif"
                            confidence = min(negative_count * 0.3 + 0.5, 0.95)
                            color = "#e74c3c"
                            emoji = "😞"
                        else:
                            predicted_sentiment = "Netral"
                            confidence = 0.6
                            color = "#f39c12"
                            emoji = "😐"
                        
                        # Tampilkan hasil
                        st.markdown("---")
                        st.markdown("#### 📊 Hasil Prediksi:")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown(f"""
                                <div style="text-align: center; padding: 2rem; background: {color}; 
                                            border-radius: 15px; color: white;">
                                    <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
                                    <h2 style="margin: 0;">{predicted_sentiment}</h2>
                                    <p style="margin: 0; font-size: 1.2rem;">Confidence: {confidence:.1%}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**Kata-kata kunci yang terdeteksi:**")
                            if positive_count > 0:
                                st.markdown(f"✅ **Positif**: {positive_count} kata")
                            if negative_count > 0:
                                st.markdown(f"❌ **Negatif**: {negative_count} kata")
                            if positive_count == 0 and negative_count == 0:
                                st.info("Tidak ada kata kunci yang terdeteksi, prediksi bersifat netral.")
                        
                        # Saran
                        st.markdown("---")
                        st.markdown("#### 💡 Saran Interpretasi:")
                        if predicted_sentiment == "Positif":
                            st.success("✅ Komentar ini menunjukkan sentimen positif terhadap Ruangguru!")
                        elif predicted_sentiment == "Negatif":
                            st.error("⚠️ Komentar ini menunjukkan sentimen negatif, perlu perhatian khusus.")
                        else:
                            st.warning("ℹ️ Komentar ini bersifat netral, tidak menunjukkan kecenderungan kuat.")
                    
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d;'>
            <p>Dashboard Analisis Sentimen Ruangguru | Dibangun dengan Streamlit & Plotly</p>
            <p>Data Source: Google Drive - Ruangguru Review Dataset</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()