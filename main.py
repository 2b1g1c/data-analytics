import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import folium
import json


def normalize_df(df):
    total_vacancies = len(df)
    low_salary_mask = (df['compensation_from'] <= 20000) | (df['compensation_to'] <= 20000) | (df['compensation_from'].isna()) | (df['compensation_to'].isna())
    low_salary_count = len(df[low_salary_mask])
    valid_vacancies = total_vacancies - low_salary_count

    low_salary_by_spec = df[low_salary_mask]['specialization'].value_counts(normalize=True) * 100
    low_salary_by_spec = low_salary_by_spec[low_salary_by_spec >= 2]
    low_salary_by_spec = pd.concat([low_salary_by_spec, pd.Series({'Другие': 100 - low_salary_by_spec.sum()})])

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.pie(
        [valid_vacancies, low_salary_count],
        labels=['Указано значение зарплаты', 'Не указано значение зарплаты'],
        colors=['#66b3ff', '#ff9999'],
        autopct='%1.1f%%'
    )
    plt.title('Доля вакансий с неуказанными зарплатами')

    custom_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
        '#98D8C8', '#F06292', '#7986CB', '#9575CD',
        '#64B5F6', '#4DB6AC', '#81C784', '#FFD54F',
        '#FA32C8', '#B03492', '#BA86CB', '#D5743D',
        '#64B5F6', '#4DB6AC', '#81C784', '#13254F',
        '#9BBCC8', '#62922A', '#7986CB', '#5BB55D',
        '#126F36', '#D4BDAB', '#45A84B', '#25A23F',
    ]

    plt.subplot(1, 2, 2)

    wedges, _, _ = plt.pie(
        low_salary_by_spec,
        labels=None,
        colors=custom_colors[:len(low_salary_by_spec)],
        autopct='%1.1f%%',
        pctdistance=0.85,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )

    plt.legend(
        wedges,
        low_salary_by_spec.index,
        title="Специализации",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    plt.title('Специализации с неуказанными зарплатами')

    plt.suptitle('Анализ вакансий с неуказанными зарплатами', fontsize=16)
    plt.tight_layout()

    plt.savefig('low_salary_analysis.png', dpi=300)
    plt.show()


def mean_median_salary_spec(df):
    df = df[df['compensation_from'] > 20000]
    df = df[df['compensation_to'] > 20000]

    plt.figure(figsize=(16, 8))


    plt.subplot(1, 2, 1)
    ax = df.groupby('specialization')['compensation_from'].mean()\
      .sort_values(ascending=False)\
      .plot(kind='bar', title='Средняя зарплата по специализациям')
    plt.ylabel('Рубли')

    plt.setp(ax.get_xticklabels(), 
            rotation=45, 
            ha='right',
            rotation_mode='anchor')

    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    # plt.savefig('mean_salary_by_specialization.png', dpi=300)
    # plt.show()


    plt.subplot(1, 2, 2)
    ax = df.groupby('specialization')['compensation_from'].median()\
      .sort_values(ascending=False)\
      .plot(kind='bar', title='Медианная зарплата по специализациям')
    plt.ylabel('Рубли')

    plt.setp(ax.get_xticklabels(), 
            rotation=45, 
            ha='right',
            rotation_mode='anchor')

    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    plt.savefig('mean_median_salary_by_specialization.png', dpi=300)
    plt.show()


def mean_median_salary_reg(df):
    df = df[df['compensation_from'] > 20000]
    df = df[df['compensation_to'] > 20000]

    plt.figure(figsize=(16, 8))


    plt.subplot(1, 2, 1)
    ax = df.groupby('region_name')['compensation_from'].mean()\
      .sort_values(ascending=False)\
      .plot(kind='bar', title='Средняя зарплата по регионам')
    plt.ylabel('Рубли')

    plt.setp(ax.get_xticklabels(), 
            rotation=45, 
            ha='right',
            rotation_mode='anchor')

    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    # plt.savefig('mean_salary_by_specialization.png', dpi=300)
    # plt.show()


    plt.subplot(1, 2, 2)
    ax = df.groupby('region_name')['compensation_from'].median()\
      .sort_values(ascending=False)\
      .plot(kind='bar', title='Медианная зарплата по регионам')
    plt.ylabel('Рубли')

    plt.setp(ax.get_xticklabels(), 
            rotation=45, 
            ha='right',
            rotation_mode='anchor')

    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    plt.savefig('mean_median_salary_by_region.png', dpi=300)
    plt.show()


def mean_median_salary_type(df):
    df = df[df['compensation_from'] > 20000]
    df = df[df['compensation_to'] > 20000]

    plt.figure(figsize=(16, 8))


    plt.subplot(1, 2, 1)
    ax = df.groupby('employment')['compensation_from'].mean()\
      .sort_values(ascending=False)\
      .plot(kind='bar', title='Средняя зарплата по типам занятости')
    plt.ylabel('Рубли')

    plt.setp(ax.get_xticklabels(), 
            rotation=45, 
            ha='right',
            rotation_mode='anchor')

    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    # plt.savefig('mean_salary_by_specialization.png', dpi=300)
    # plt.show()


    plt.subplot(1, 2, 2)
    ax = df.groupby('employment')['compensation_from'].median()\
      .sort_values(ascending=False)\
      .plot(kind='bar', title='Медианная зарплата по типам занятости')
    plt.ylabel('Рубли')

    plt.setp(ax.get_xticklabels(), 
            rotation=45, 
            ha='right',
            rotation_mode='anchor')

    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    plt.savefig('mean_median_salary_by_type.png', dpi=300)
    plt.show()


def break_min_max_spec(df):
    df = df[df['compensation_from'] > 20000]
    df = df[df['compensation_to'] > 20000]

    plt.figure(figsize=(16, 8))
    plt.ticklabel_format(axis='y', style='plain', useOffset=False)

    ax = (df.groupby('specialization')['compensation_to'].max() - 
          df.groupby('specialization')['compensation_from'].min())\
      .sort_values(ascending=False)\
      .plot(kind='bar', title='Разрыв между макс и мин зарплатой по специализациям')
    plt.ylabel('Рубли')

    plt.setp(ax.get_xticklabels(), 
            rotation=45, 
            ha='right',
            rotation_mode='anchor')

    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    plt.savefig('break_salary_by_specialization.png', dpi=300)
    plt.show()


def break_min_max_reg(df):
    df = df[df['compensation_from'] > 20000]
    df = df[df['compensation_to'] > 20000]

    plt.figure(figsize=(16, 8))
    plt.ticklabel_format(axis='y', style='plain', useOffset=False)

    ax = (df.groupby('region_name')['compensation_to'].max() - 
          df.groupby('region_name')['compensation_from'].min())\
      .sort_values(ascending=False)\
      .plot(kind='bar', title='Разрыв между макс и мин зарплатой по регионам')
    plt.ylabel('Рубли')

    plt.setp(ax.get_xticklabels(), 
            rotation=45, 
            ha='right',
            rotation_mode='anchor')

    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    plt.savefig('break_salary_by_region.png', dpi=300)
    plt.show()


def break_min_max_type(df):
    df = df[df['compensation_from'] > 20000]
    df = df[df['compensation_to'] > 20000]

    plt.figure(figsize=(16, 8))
    plt.ticklabel_format(axis='y', style='plain', useOffset=False)

    ax = (df.groupby('employment')['compensation_to'].max() - 
          df.groupby('employment')['compensation_from'].min())\
      .sort_values(ascending=False)\
      .plot(kind='bar', title='Разрыв между макс и мин зарплатой по типам занятости')
    plt.ylabel('Рубли')

    plt.setp(ax.get_xticklabels(), 
            rotation=45, 
            ha='right',
            rotation_mode='anchor')

    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    plt.savefig('break_salary_by_type.png', dpi=300)
    plt.show()



def employees_number(df):
    df_clean = df[df['employees_number'] > 0]
    df_clean = df_clean.drop_duplicates('employer_id')

    bins = [1, 50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000, 500000, np.inf]
    labels = ['1-50', '51-100', '101-200', '201-500', 
              '501-1k', '1k-5k', '5k-10k', '10k-50k', '50k-100k', '100k-500k', '500k+']

    df_clean['size_group'] = pd.cut(df_clean['employees_number'], bins=bins, labels=labels)
    size_counts = df_clean['size_group'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    ax = plt.subplot()

    size_counts.plot(
        kind='bar',
        color='skyblue',
        edgecolor='white',
        ax=ax
    )

    ax.set_title('Количество компаний относительно размера штата', pad=20)
    ax.set_xlabel('Диапазон численности сотрудников')
    ax.set_ylabel('Количество компаний')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for p in ax.patches:
        ax.annotate(f"{p.get_height():,.0f}".replace(",", " "), 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig('company_size_amount.png', dpi=300, bbox_inches='tight')
    plt.show()


def salary_responses_about_company_size(df):
    df_clean = df[df['employees_number'] > 0]

    bins = [1, 50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000, 500000, np.inf]
    labels = ['1-50', '51-100', '101-200', '201-500', 
                '501-1k', '1k-5k', '5k-10k', '10k-50k', '50k-100k', '100k-500k', '500k+']

    plt.figure(figsize=(18, 6))

    ax1 = plt.subplot(1, 2, 1)

    df_clean['size_group'] = pd.cut(df_clean['employees_number'], bins=bins, labels=labels)

    salary_by_size = df_clean.groupby('size_group')['compensation_from'].median()

    salary_by_size.plot(
        kind='bar',
        color='#66b3ff',
        edgecolor='white',
        ax=ax1
    )

    ax1.set_title('Медианная зарплата по размеру компании', pad=20)
    ax1.set_xlabel('Численность сотрудников')
    ax1.set_ylabel('Зарплата "от" (рубли)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():,.0f}".replace(",", " "), 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points')

    ax2 = plt.subplot(1, 2, 2)

    responses_by_size = df_clean.groupby('size_group')['response_count'].median()

    responses_by_size.plot(
        kind='bar',
        color='#99ff99',
        edgecolor='white',
        ax=ax2
    )

    ax2.set_title('Медианное количество откликов по размеру компании', pad=20)
    ax2.set_xlabel('Численность сотрудников')
    ax2.set_ylabel('Количество откликов')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.0f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig('company_size_salary_and_responses.png')
    plt.show()


def monthly_vacanciess(df):
    df['creation_date'] = pd.to_datetime(df['creation_date'])

    month_names = {
        1: 'Январь', 2: 'Февраль', 3: 'Март',
        4: 'Апрель', 5: 'Май', 6: 'Июнь',
        7: 'Июль', 8: 'Август', 9: 'Сентябрь',
        10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
    }

    df['month_name'] = df['creation_date'].dt.month.map(month_names)
    monthly_trend = df.groupby('month_name').size()
    monthly_trend = monthly_trend.reindex(list(month_names.values())) 
    plt.figure(figsize=(14, 6))
    ax = plt.subplot()

    monthly_trend.plot(
        kind='line',
        markersize=8,
        linewidth=2,
        color='#4e79a7',
        ax=ax
    )
    ax.set_title('Динамика создания вакансий по месяцам', pad=20, fontsize=14)
    ax.set_xlabel('Месяц')
    ax.set_ylabel('Количество вакансий')
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.set_xticks(range(len(month_names)))
    ax.set_xticklabels(month_names.values())

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('monthly_vacancies_trend.png')
    plt.show()


def split_responses(row):
    creation_date = row['creation_date']
    end_date = creation_date + timedelta(days=30)
    response_count = row['response_count']    
    if creation_date.month == end_date.month:
        return {creation_date.strftime('%B'): response_count}    
    days_in_first_month = (end_date.replace(day=1) - creation_date).days
    first_month = creation_date.strftime('%B')
    second_month = end_date.strftime('%B')    
    return {
        first_month: int(response_count * days_in_first_month / 30),
        second_month: response_count - int(response_count * days_in_first_month / 30)
    }


def montly_responses(df):
    df['creation_date'] = pd.to_datetime(df['creation_date'])
    month_names = [
        'Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
        'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь'
    ]
    month_en = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    responses_distributed = []
    for _, row in df.iterrows():
        month_counts = split_responses(row)
        for month, count in month_counts.items():
            responses_distributed.append({'month': month, 'count': count})
    responses_df = pd.DataFrame(responses_distributed)

    monthly_responses = responses_df.groupby('month')['count'].sum().reindex(month_en)

    plt.figure(figsize=(14, 6))
    ax = monthly_responses.plot(
        kind='line',
        color='#e15759',
        linewidth=2
    )

    ax.set_title('Распределение откликов по месяцам', pad=20)
    ax.set_ylabel('Количество откликов')
    ax.set_xlabel('Месяца')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_xticks(range(len(month_names)))
    ax.set_xticklabels(month_names, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('monthly_responses_trend.png')
    plt.show()

    # print(responses_df['count'].sum())
    # print(df['response_count'].sum())


def vacancies_and_responses_by_spec(df):
    vacancies_by_spec = df['specialization'].value_counts()

    responses_by_spec = df.groupby('specialization')['response_count'].sum()

    plt.figure(figsize=(12, 10))
    plt.ticklabel_format(axis='y', style='plain', useOffset=False)

    vacancies_by_spec.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title('Количество вакансий по специальностям', pad=20, fontsize=14)
    plt.xlabel('Специальность')
    plt.ylabel('Количество вакансий')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('vacancies_by_spec.png')
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.ticklabel_format(axis='y', style='plain', useOffset=False)

    responses_by_spec.sort_values(ascending=False).plot(kind='bar', color='salmon')
    plt.title('Количество откликов по специальностям', pad=20, fontsize=14)
    plt.xlabel('Специальность')
    plt.ylabel('Количество откликов')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('responses_by_spec.png')
    plt.show()


def monthly_largest_specs_dynamic(df):
    df['creation_date'] = pd.to_datetime(df['creation_date'])
    df['month'] = df['creation_date'].dt.month

    month_names = {
        1: 'Янв', 2: 'Фев', 3: 'Мар', 4: 'Апр', 5: 'Май', 6: 'Июн',
        7: 'Июл', 8: 'Авг', 9: 'Сен', 10: 'Окт', 11: 'Ноя', 12: 'Дек'
    }

    top5_spec_vacancies = df['specialization'].value_counts().nlargest(5).index.tolist()

    top5_spec_responses = df.groupby('specialization')['response_count'].sum().nlargest(5).index.tolist()

    def prepare_monthly_data(df, top_specs, mode='vacancies'):
        result = pd.DataFrame(index=top_specs, columns=sorted(df['month'].unique()))
        for month in sorted(df['month'].unique()):
            month_data = df[df['month'] == month]
            for spec in top_specs:
                if mode == 'vacancies':
                    count = (month_data['specialization'] == spec).sum()
                else:
                    count = month_data[month_data['specialization'] == spec]['response_count'].sum()
                result.at[spec, month] = count if not pd.isna(count) else 0
        return result.fillna(0)

    monthly_vacancies = prepare_monthly_data(df, top5_spec_vacancies, 'vacancies')
    monthly_responses = prepare_monthly_data(df, top5_spec_responses, 'responses')

    plt.figure(figsize=(10, 6))

    for spec in monthly_vacancies.index:
        plt.plot([month_names[m] for m in monthly_vacancies.columns], 
                monthly_vacancies.loc[spec], label=spec, linewidth=2)

    plt.title('Динамика топ-5 специальностей по вакансиям', 
            fontsize=14, pad=20)
    plt.xlabel('Месяц', fontsize=12)
    plt.ylabel('Количество вакансий', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig('5_largest_specs_by_vacancies_monthly.png')
    plt.show()


    plt.figure(figsize=(10, 6))
    for spec in monthly_responses.index:
        plt.plot([month_names[m] for m in monthly_responses.columns], 
                monthly_responses.loc[spec], label=spec, linewidth=2)

    plt.title('Динамика топ-5 специальностей по откликам', 
            fontsize=14, pad=20)
    plt.xlabel('Месяц', fontsize=12)
    plt.ylabel('Количество откликов', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig('5_largest_specs_by_responses_monthly.png')
    plt.show()


def style_function(feature):
    return {
        "fillColor": "YlOrRd",
        "color": "black",
        "weight": 1,
        "fillOpacity": 0,
    }


def map_vacancies(m, geojson_data):
    geojson = folium.GeoJson(
        geojson_data,
        name="Регионы",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "vacancies"],
            aliases=["Регион:", "Количество вакансий:"],
            localize=True
        )
    )
    folium.Choropleth(
        geo_data=geojson_data,
        data=regions_stats,
        columns=["region_name", "vacancy_id"],
        key_on="feature.properties.name",
        fill_color="YlOrRd",
        legend_name="Количество вакансий"
    ).add_to(m)
    geojson.add_to(m)
    m.save("vacancies_map.html")


def map_salary(m, geojson_data):
    geojson = folium.GeoJson(
        geojson_data,
        name="Регионы",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "med_salary"],
            aliases=["Регион:", "Медианная зарплата:"],
            localize=True
        )
    )
    folium.Choropleth(
        geo_data=geojson_data,
        data=regions_stats,
        columns=["region_name", "vacancy_id"],
        key_on="feature.properties.name",
        fill_color="PuOr",
        legend_name="Медианная зарплата"
    ).add_to(m)
    geojson.add_to(m)
    m.save("salary_map.html")


def map_responses(m, geojson_data):
    geojson = folium.GeoJson(
        geojson_data,
        name="Регионы",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "responses"],
            aliases=["Регион:", "Количество откликов:"],
            localize=True
        )
    )
    folium.Choropleth(
        geo_data=geojson_data,
        data=regions_stats,
        columns=["region_name", "vacancy_id"],
        key_on="feature.properties.name",
        fill_color="RdBu",
        legend_name="Количество откликов"
    ).add_to(m)
    geojson.add_to(m)
    m.save("responses_map.html")


def normalize_name(name):
    return (name.replace("Еврейская АО", "Еврейская автономная область")
            .replace("АО", "автономный округ")
            .replace("Республика ", "")
            .replace("Саха (Якутия)", "Республика Саха (Якутия)")
            .replace("Хакасия", "Республика Хакасия")
            .strip())


# функция анализа опыта работы (5) и его влияния на различные сферы
def analyze_experience(df):
    # для замены значений на русские
    experience_dict = {
        'no_experience': 'Нет опыта',
        'up_to_3_years': 'От 1 года до 3 лет',
        'up_to_6_years': 'От 3 до 6 лет',
        'above_6_years': 'Более 6 лет'
    }
    
    # порядок категорий для правильного отображения
    experience_order = [
        'Нет опыта',
        'От 1 года до 3 лет',
        'От 3 до 6 лет',
        'Более 6 лет'
    ]
    
    # заменяем значения на русские
    df['experience'] = df['length_of_employment'].map(experience_dict)
    
    # убираем строки с неизвестными категориями опыта
    df_exp = df[df['experience'].notna()]
    
    # преобразуем в тип с правильным порядком
    df_exp['experience'] = pd.Categorical(
        df_exp['experience'], 
        categories=experience_order,
        ordered=True
    )
    
    # анализ частоты требований к опыту
    plt.figure(figsize=(12, 6))
    experience_counts = df_exp['experience'].value_counts().loc[experience_order]
    
    ax = experience_counts.plot(
        kind='bar', 
        color='#4e79a7',
        edgecolor='white'
    )
    
    plt.title('Распределение требований к опыту работы', fontsize=16, pad=20)
    plt.xlabel('Категория опыта', fontsize=12)
    plt.ylabel('Количество вакансий', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # добавление значений над столбцами
    for i, v in enumerate(experience_counts):
        ax.text(i, v + 20, f"{v:,}".replace(',', ' '), 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('experience_distribution.png', dpi=300)
    plt.show()

    # фильтрация вакансий с указанной зп
    salary_df = df_exp[
        (df_exp['compensation_from'] > 20000) & 
        (df_exp['compensation_to'] > 20000)
    ].copy()
    
    # влияние опыта на зп и приглашения
    plt.figure(figsize=(14, 6))
    
    # график Медианной зп по опыту
    plt.subplot(1, 2, 1)
    salary_by_exp = salary_df.groupby('experience')['compensation_from'].median()
    ax1 = salary_by_exp.loc[experience_order].plot(
        kind='bar', 
        color='#f28e2c',
        edgecolor='white'
    )
    plt.title('Медианная зарплата по опыту работы', fontsize=15)
    plt.xlabel('Категория опыта', fontsize=12)
    plt.ylabel('Рубли', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # жобавление значений на столбцы зп
    for i, v in enumerate(salary_by_exp.loc[experience_order]):
        ax1.text(i, v + 500, f"{v:,.0f} ₽".replace(',', ' '), 
                ha='center', fontsize=10)
    

    # график средних приглашений по опыту
    plt.subplot(1, 2, 2)
    invitations_by_exp = df_exp.groupby('experience')['invitation_count'].median()
    ax2 = invitations_by_exp.loc[experience_order].plot(
        kind='bar', 
        color='#59a14f',
        edgecolor='white'
    )
    plt.title('Среднее количество приглашений  по опыту работы', fontsize=13)
    plt.xlabel('Категория опыта', fontsize=12)
    plt.ylabel('Количество приглашений', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # добавление значений на столбцы приглашений
    for i, v in enumerate(invitations_by_exp.loc[experience_order]):
        ax2.text(i, v + 0.1, f"{v:.1f}", 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('experience_salary_invitations.png', dpi=300)
    plt.show()
    
    # смотрим зп vs приглашения 
    # нормализуем данные для совмещения на одном графике (смотрим относительно максимума)
    norm_salary = salary_by_exp / salary_by_exp.max()
    norm_invitations = invitations_by_exp / invitations_by_exp.max()
    
    # создаем DataFrame для удобства построения
    combined = pd.DataFrame({
        'Зарплата (нормализованная)': norm_salary.loc[experience_order],
        'Приглашения (нормализованная)': norm_invitations.loc[experience_order]
    })
    
    ax = combined.plot(kind='bar', figsize=(12, 6))
    plt.title('Сравнение зарплат и приглашений по опыту (нормализованное)', fontsize=16, pad=20)
    plt.xlabel('Категория опыта', fontsize=12)
    plt.ylabel('Нормализованные значения', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('experience_salary_vs_invitations.png', dpi=300)
    plt.show()

# функция для гендерного и возрастного анализа (6)
def analyze_gender_age(df):
    # сравнение откликов по полу
    total_female_responses = df['female_response_count'].sum()
    total_male_responses = df['male_response_count'].sum()
    
    # сравнение приглашений по полу
    total_female_invitations = df['female_invitation_count'].sum()
    total_male_invitations = df['male_invitation_count'].sum()
    
    # визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # отклики
    ax1.pie([total_female_responses, total_male_responses],
            labels=['Женщины', 'Мужчины'],
            autopct='%1.1f%%',
            colors=['#ff9da7', '#86b4e8'])
    ax1.set_title('Распределение откликов по полу')
    
    # приглашения
    ax2.pie([total_female_invitations, total_male_invitations],
            labels=['Женщины', 'Мужчины'],
            autopct='%1.1f%%',
            colors=['#ff9da7', '#86b4e8'])
    ax2.set_title('Распределение приглашений по полу')
    
    plt.savefig('gender_distribution.png', dpi=300)
    plt.show()
    
    # анализ молодых соискателей
    teen_df = df[df['accept_teenagers'] == True]
    
    # распределение вакансий для подростков
    plt.figure(figsize=(12, 6))
    teen_spec_counts = teen_df['specialization'].value_counts().nlargest(10)
    ax = teen_spec_counts.plot(kind='bar', color='#76b7b2')
    plt.title('Топ-10 специализаций для соискателей 14-18 лет')
    plt.xlabel('Специализация')
    plt.ylabel('Количество вакансий')
    plt.xticks(rotation=45, ha='right')
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,}".replace(',', ' '), 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('teen_vacancies.png', dpi=300)
    plt.show()
    
    # активность
    total_teen_responses = df['young_response_count'].sum()
    total_teen_invitations = df['young_invitation_count'].sum()
    
    print(f"Общее количество откликов от молодежи: {total_teen_responses}")
    print(f"Общее количество приглашений молодежи: {total_teen_invitations}")

# функция для анализа отраслей и Зп (7)
def analyze_industries(df):

    # словарь отраслей
    industry_dict = {
        5: "Логистика",
        7: "IT",
        8: "Электроника",
        9: "Телекоммуникации",
        11: "Маркетинг",
        13: "Строительство",
        15: "Автобизнес",
        19: "Лесная промышленность",
        24: "Металлургия",
        27: "Продукты питания",
        29: "Сельское хозяйство",
        33: "Машиностроение",
        34: "Химическая пром.",
        36: "Госорганизации",
        37: "НКО",
        39: "Образование",
        41: "Розничная торговля",
        42: "Товары народного потребления",
        43: "Финансы",
        44: "Услуги для бизнеса",
        45: "Добывающая отрасль",
        46: "Энергетика",
        47: "Нефть и газ",
        48: "Медицина",
        49: "Услуги населению",
        50: "HoReCa",
        51: "ЖКХ",
        52: "Культура",
        388: "Промышленное оборудование",
        389: "Управление активами"
    }
    
    # функция для анализа строки 
    def parse_industries(industry_str):
        # отсутствующие значения
        if not isinstance(industry_str, str):
            return []
        
        # удаление квадратных скобок и пробелов
        clean_str = industry_str.strip('[]').replace(' ', '')
        if not clean_str:
            return []
        
        # разделение строки по запятым
        industry_ids = clean_str.split(',')
        industries = []
        
        for i in industry_ids:
            try:
                # преобразование в целое число и поиск в словаре
                industry_id = int(i)
                industry_name = industry_dict.get(industry_id)
                if industry_name:
                    industries.append(industry_name)
                else:
                    # добавление неизвестных ID с пометкой
                    industries.append(f"Unknown ID: {industry_id}")
            except ValueError:
                # сохранение некорректных значений
                industries.append(f"Invalid value: '{i}'")
        
        return industries
    
    # функция преобразования
    df['industries'] = df['industry_id_list'].apply(parse_industries)
    
    # подсчет вакансий по отраслям
    industry_counts = {}
    for industries in df['industries']:
        for industry in industries:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
    
    # сортировка и преобразование в Series
    industry_counts = pd.Series(industry_counts).sort_values(ascending=False)
    
    # визуализация распределения отраслей (только известных)
    known_industries = industry_counts[~industry_counts.index.str.contains('Unknown|Invalid')]
    
    plt.figure(figsize=(14, 8))
    ax = known_industries.head(15).sort_values().plot(kind='barh', color='#59a14f')
    plt.title('Топ-15 отраслей по количеству вакансий', fontsize=14)
    plt.xlabel('Количество вакансий', fontsize=12)
    plt.ylabel('Отрасль', fontsize=12)
    
    # добавление значений на столбцы
    for i, v in enumerate(known_industries.head(15).sort_values()):
        ax.text(v + 5, i, f"{v:,}".replace(',', ' '), va='center')
    
    plt.tight_layout()
    plt.savefig('industry_distribution.png', dpi=300)
    plt.show()
    
    # анализ зп по отраслям (только для вакансий с указанной зп)
    salary_df = df[(df['compensation_from'] > 20000) & (df['compensation_to'] > 20000)].copy()
    
    # сбор зп по отраслям
    industry_salaries = {}
    for idx, row in salary_df.iterrows():
        salary = row['compensation_from']
        for industry in row['industries']:
            # пропускаем неизвестные и некорректные отрасли
            if 'Unknown' in industry or 'Invalid' in industry:
                continue
            if industry not in industry_salaries:
                industry_salaries[industry] = []
            industry_salaries[industry].append(salary)
    
    # расчет Медианных зп
    median_salaries = {ind: np.median(sals) for ind, sals in industry_salaries.items()}
    median_salaries = pd.Series(median_salaries).sort_values(ascending=False)
    
    # визуализация зарплат по отраслям
    plt.figure(figsize=(14, 8))
    ax = median_salaries.head(15).sort_values().plot(kind='barh', color='#edc949')
    plt.title('Топ-15 отраслей по медианной зарплате', fontsize=14)
    plt.xlabel('Медианная зарплата (руб)', fontsize=12)
    plt.ylabel('Отрасль', fontsize=12)
    
    # добавление значений на столбцы
    for i, v in enumerate(median_salaries.head(15).sort_values()):
        ax.text(v + 500, i, f"{v:,.0f} ₽".replace(',', ' '), va='center')
    
    plt.tight_layout()
    plt.savefig('industry_salaries.png', dpi=300)
    plt.show()
    
    # вывод статистики по неизвестным отраслям
    unknown_industries = industry_counts[industry_counts.index.str.contains('Unknown')]
    if not unknown_industries.empty:
        print("\nОбнаружены неизвестные ID отраслей:")
        for industry, count in unknown_industries.items():
            print(f"- {industry}: {count} вакансий")
    
    invalid_industries = industry_counts[industry_counts.index.str.contains('Invalid')]
    if not invalid_industries.empty:
        print("\nОбнаружены некорректные значения в industry_id_list:")
        for industry, count in invalid_industries.items():
            print(f"- {industry}: {count} вакансий")


df = pd.read_csv('bd_hh.csv')


# Преобразование данных
df['creation_date'] = pd.to_datetime(df['creation_date'])
    

# блок зарплаты

normalize_df(df)
mean_median_salary_spec(df)
break_min_max_spec(df)
mean_median_salary_reg(df)
break_min_max_reg(df)
mean_median_salary_type(df)
break_min_max_type(df)
 
print("\nАнализ опыта работы:")
analyze_experience(df)
    
print("\nГендерный и возрастной анализ:")
analyze_gender_age(df)
    
print("\nАнализ отраслей:")
analyze_industries(df)

 # интересные данные
 
print("Количество уникальных компаний:", df[df['employees_number'] > 0]['employer_id'].nunique())
print("Общее количество сотрудников во всех компаниях:", df[df['employees_number'] > 0].drop_duplicates('employer_id')['employees_number'].sum())
 
print("РЖД:")
specializations = df[df['employees_number'] == df['employees_number'].max()]['specialization']
print(specializations.value_counts())
 
 
 # блок численности штата
 
employees_number(df)
salary_responses_about_company_size(df)
 
 
 # блок временных рядов
 
monthly_vacanciess(df)
montly_responses(df)
 
monthly_largest_specs_dynamic(df)
vacancies_and_responses_by_spec(df)


# блок карт

regions_stats = df.groupby('region_name').agg({
    'vacancy_id': 'count',
    'compensation_from': 'median',
    'response_count': 'sum'
}).reset_index()

regions_stats['region_name'] = regions_stats['region_name'].apply(normalize_name)

with open("russia.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

stats_dict = regions_stats.set_index('region_name').to_dict('index')

for feature in geojson_data['features']:
    region_name = feature['properties']['name']
    stats = stats_dict.get(region_name, {})
    feature['properties']['vacancies'] = f"{stats.get('vacancy_id', 0):,}".replace(",", " ")
    feature['properties']['med_salary'] = f"{round(stats.get('compensation_from', 0), 1):,}".replace(",", " ") + " ₽"
    feature['properties']['responses'] = f"{stats.get('response_count', 0):,}".replace(",", " ")


m = folium.Map(location=[60, 80], zoom_start=3)
map_vacancies(m, geojson_data)
m = folium.Map(location=[60, 80], zoom_start=3)
map_salary(m, geojson_data)
m = folium.Map(location=[60, 80], zoom_start=3)
map_responses(m, geojson_data)
