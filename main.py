import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



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


    # Добавляем легенду с цветами
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

    # plt.style.use('seaborn-v0_8')
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

    # plt.style.use('seaborn-v0_8')
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

    # plt.style.use('seaborn-v0_8')
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







df = pd.read_csv('bd_hh.csv')
normalize_df(df)
mean_median_salary_spec(df)
break_min_max_spec(df)
mean_median_salary_reg(df)
break_min_max_reg(df)
mean_median_salary_type(df)
break_min_max_type(df)


print("Количество уникальных компаний:", df[df['employees_number'] > 0]['employer_id'].nunique())
print("Общее количество сотрудников во всех компаниях:", df[df['employees_number'] > 0].drop_duplicates('employer_id')['employees_number'].sum())

print("РЖД:")
specializations = df[df['employees_number'] == df['employees_number'].max()]['specialization']
print(specializations.value_counts())

employees_number(df)
salary_responses_about_company_size(df)

