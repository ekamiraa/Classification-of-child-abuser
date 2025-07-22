import streamlit as st
import pandas as pd
from pipeline import df, pipeline_from_row

#### ----------------------- Page 1 (Home) ----------------------------- #####
def page1():
    st.title("👨‍✈️: Identification of Child Abuser.")
    multi = """
        This is a simple text classification web application to identify abusers from child abuse news using neural network,  
        by :blue-background[Eka Mira Novita Subroto]. &mdash; 👨📄🧠
        """
    st.write(multi)

    options = [(i, df['Berita'][i][:100] + '...') for i in df.index]
    selected_option = st.selectbox(
        "Select news from the dataset:",
        options,
        index=None,
        placeholder="Select the news...",
        format_func=lambda x: x[1] if x is not None else "Select..."
    )

    if selected_option is not None:
        selected_index = selected_option[0]
        
        # Ambil teks berita dari index yang dipilih
        selected_text = df['Berita'][selected_index]

        # Tampilkan isi berita di dalam text_area
        txt = st.text_area("News text:", value=selected_text, height=300)

        # st.write(f"Index of the selected news: {selected_index}")
        st.write(f"You wrote {len(txt)} characters.")

        if st.button("Process"):
            with st.spinner("Process..."):
                st.session_state["result"] = pipeline_from_row(selected_index)

        # Setelah tombol "Process", cek apakah result sudah ada
        if "result" in st.session_state:
            data = st.session_state["result"]["data"]

            st.subheader("Category Prediction")
            labels = ['Keluarga', 'Pengasuh', 'Tenaga Pendidik', 'Teman', 'Orang Asing']
            st.success(f"Prediction Result: **{labels[data['prediction']]}**")

            if st.button("See more...."):
                st.subheader("🔧 Text Preprocessing")
                st.write(data["preprocessed"])

                st.subheader("🔢 Word2Vec Vector")
                st.write(str(data['vector']))

                st.subheader("🎯 Prediction Probability of Each Category")
                softmax_output = data['nn_output']['q_activation']
                for i, prob in enumerate(softmax_output):
                    st.write(f"**{labels[i]}:** {round(prob[0]*100, 2)}%")                    
    else:
        st.info("Please select a news item to continue.")


#### ----------------------- Page 2 (Dataset) ----------------------------- #####
def page2():
    st.title("📰: Child Abuse News Dataset.")
    multi = """
        The dataset comes from Indonesian online news portals:  
        :blue-background[Detik, CNN, TribunNews, Kompas, and Republika] 
        from January 2023 to September 2024 &mdash; 🔗⌨️📑
        """
    st.write(multi)

    # Load preprocessed data dan vektor dokumen
    df = pd.read_csv('Result/All_Process/All_Process.csv')  # termasuk kolom 'Berita' dan 'word2vec_vector'
    print("DataFrame Loaded")

    # Load preprocessed data dan vektor dokumen 
    st.dataframe(df[['berita', 'label', 'category_id']], use_container_width=True, hide_index=True)


#### ----------------------- Page 3 (Research) ----------------------------- #####
def page3():
    st.title("📰: Identifying The Relationship between Victims and Child Abusers based on Text Classification Using Neural Network.")
    multi = """
        This is a simple text classification web application to identify abusers from child abuse news using neural network,  
        by :blue-background[Eka Mira Novita Subroto]. &mdash; 👨📄🧠
        """
    st.write(multi)

    st.subheader("👓 Background")
    background = """
        Cases of child abuse in Indonesia continue to increase each year and have become a serious concern. The impact of abuse is not only 
        physical but also affects the child's psychological and social development. Many of these acts are committed by people close to the 
        child, such as family members, caregivers, teachers, or friends. Identifying the relationship between the victim and the perpetrator 
        is essential to understanding the patterns and causes of abuse, so that better prevention efforts can be made. Online news platforms 
        often report child abuse incidents and can be used as a valuable data source. However, manually identifying perpetrators from these 
        news articles is difficult and time-consuming. Therefore, this study uses text classification with a neural network approach to 
        automatically detect the type of perpetrator based on the news content. This method is expected to help authorities and child protection 
        institutions respond more quickly and effectively to abuse cases.
        """
    st.write(background)

    st.subheader("🔑 Objective")
    objective = """
        To evaluate the performance of neural network-based text classification in identifying the relationship between child abuse victims 
        and perpetrators using online news data. A total of nine neural network configurations were developed by combining three different 
        values for the number of nodes in the hidden layer (10, 30, and 60 nodes) with three learning rate values (0.001, 0.005, and 0.01).
        These parameters were chosen to observe how changes in model complexity and learning speed affect performance in different evaluation 
        stages. Each configuration was labeled accordingly (e.g., X1, X2, ..., Z3)
        """
    st.write(objective)

    st.subheader("🧾 Dataset")
    dataset = """
        The dataset comprises 975 manually labeled child abuse news articles collected through web scraping from Indonesian online news portals, 
        including Detik, TribunNews, CNN, Kompas, and Republika. The data spans from January 2023 to September 2024.
        """
    st.write(dataset)

    st.subheader("🧩 Categories")
    st.markdown("""
    - Family
    - Caregiver
    - Educator
    - Friend
    - Stranger
    """)

    st.subheader("📈 Result")
    df = pd.DataFrame(
    {
        "Evaluation Phase": ["Training", "Testing", "K-Fold Validation"],
        "Performance Matrix": ["Lowest cost and fastest time", "Evaluation Matrix", "Average accuracy and standar deviation"],
        "Best Model": ["Model Z3", "Model Y2", "Model Z2"],
        "Value": ["Cost = 0.0484, Epoch = 26172, Time = 139,76s", "Accuracy = 0.805, F1-Score = 0,745", "Average accuracy = 0.829, Standar Deviation = 0.0306"],
    }
    )
    st.dataframe(df, hide_index=True)
    result = """
        This study evaluates the performance of a neural network model through three different phases: training, testing, and k-fold cross-validation. 
        Each phase uses different performance indicators. During the training phase, the evaluation focused on the cost reduction over epochs and 
        training time (in seconds). The best model in this phase, Model Z3 (node = 60, learning rate = 0.01), showed the most efficient learning process with the fastest convergence 
        and lowest final cost. In the testing phase, model performance was measured using standard classification metrics—accuracy, precision, recall, 
        and F1-score. Model Y2 (node = 30, learning rate = 0.005) performed best in this phase with the highest accuracy of 80,50%, indicating its strong capability in generalizing to
        unseen data. Lastly, in the k-fold cross-validation phase, the focus was on the model’s stability and consistency, measured by the average 
        accuracy across all folds. Model Z2 (node = 60, learning rate = 0.005) achieved the highest average accuracy of 82,90%, demonstrating robust performance across different 
        subsets of the dataset. These three models represent different strengths: learning efficiency (Z3), generalization (Y2), and consistency (Z2).
        """
    st.write(result)

    st.subheader("📝 Word Analysis per Category")
    st.markdown("""
    - **Family**
    """)
    st.image("Result/WordCloud/keluarga.png", caption="Wordcloud for family category", width=200)
    family = """
        In the family category, words such as father ("ayah"), biological ("kandung"), and step ("tiri") appear dominantly, indicating the direct 
        involvement of parents in child abuse cases. Other prominent terms like rape ("perkosa"), molestation ("cabul"), violence ("keras"), assault 
        ("aniaya"), and threat ("ancam") suggest that the abuse is often physical and sexual in nature. The word home ("rumah") also appears frequently, 
        implying that such abuse typically occurs within the household environment.
        """
    st.write(family)
    st.markdown("""
    - **Caregiver**
    """)
    st.image("Result/WordCloud/pengasuh.png", caption="Wordcloud for caregiver category", width=200)
    caregiver = """
        The caregiver category often involves cases related to daycare centers or child care facilities. Words such as abuse ("aniaya") and care ("asuh") 
        are prevalent, suggesting that physical abuse in this category is usually the result of negligence or rough treatment from those responsible for 
        child supervision.
        """
    st.write(caregiver)
    st.markdown("""
    - **Educator**
    """)
    st.image("Result/WordCloud/tenaga pendidik.png", caption="Wordcloud for educator category", width=200)
    educator = """
        The most dominant words are teacher ("guru"), student ("siswa"), and school ("sekolah"), clearly pointing to an educational context. Additionally, 
        words like molestation ("cabul") and sexual ("seksual") frequently appear, indicating that the most common type of abuse by educators involves 
        sexual harassment committed by teachers or school staff toward students.
        """
    st.write(educator)
    st.markdown("""
    - **Friend**
    """)
    st.image("Result/WordCloud/teman.png", caption="Wordcloud for friend category", width=200)
    friend = """
        The friend category is characterized by terms related to the school environment, including school ("sekolah"), student ("siswa"), bully ("rundung"), 
        and bullying ("bullying"). These terms indicate that peer violence often takes the form of school bullying. The word video ("video") also stands out, 
        implying that many of these incidents were recorded and went viral on social media platforms.
        """
    st.write(friend)
    st.markdown("""
    - **Stranger**
    """)
    st.image("Result/WordCloud/orang asing.png", caption="Wordcloud for stranger category", width=200)
    stranger = """
        The dominant words include molestation ("cabul") and man ("pria"), indicating a pattern of sexual abuse committed by unknown male perpetrators. The word
        video ("video") also appears frequently, suggesting that some of these cases gained public attention after being captured on video recordings.
        """
    st.write(stranger)

pg = st.navigation([
    st.Page(page1, title="Home", icon="🏠"),
    st.Page(page2, title="Dataset", icon="📄"),
    st.Page(page3, title="Research", icon="📊")
])
pg.run()

