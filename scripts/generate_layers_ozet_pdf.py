"""Mor temalı src/layers.py özet PDF üretici."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

# Mor palet
PURPLE_DARK = "#4A148C"
PURPLE_MID = "#7B1FA2"
PURPLE_LIGHT = "#E1BEE7"
PURPLE_ACCENT = "#CE93D8"
WHITE = "#FFFFFF"
TEXT_DARK = "#2D0A45"

OUTPUT = Path(__file__).resolve().parent.parent / "layers_ozet.pdf"

SUMMARY_BLOCKS = [
    (
        "Dosya amacı",
        "TensorFlow/Keras için özel dikkat (attention) katmanları tanımlar. "
        "Özellikle kanal ve uzamsal dikkat modüllerini (CBAM benzeri) "
        "video/görüntü modellerinde kullanılmak üzere sunar.",
    ),
    (
        "ChannelAttention",
        "Kanallar arası ilişkiyi modeller (hangi kanallar önemli).\n"
        "• Varsayılan ratio=8\n"
        "• Akış: Global Average Pooling → Dense(relu, channel/ratio) → Dense(channel) → Sigmoid\n"
        "• Giriş ile aynı boyutta çıktı: inputs * scale\n"
        "• build(): alt Dense katmanları açıkça build edilir (ağırlık yükleme için)\n"
        "• get_config(): ratio serileştirmeye dahil",
    ),
    (
        "SpatialAttention",
        "Uzamsal olarak «neresi önemli» sorusuna odaklanır.\n"
        "• Varsayılan kernel_size=7\n"
        "• Kanal ekseninde ortalama ve maksimum havuzlama → birleştirme (2 kanal)\n"
        "• Conv2D(1 filtre, same padding, sigmoid) → inputs * scale\n"
        "• get_config(): kernel_size serileştirmeye dahil",
    ),
    (
        "Ortak özellikler",
        "• tensorflow.keras.layers.Layer alt sınıfları\n"
        "• He normal başlatma; çıktı dtype giriş dtype ile uyumlu (tf.cast)\n"
        "• compute_output_shape: şekil değişmez\n"
        "• Projede muhtemel kullanım: src/model.py içinde özellik haritalarına dikkat ekleme",
    ),
]


def draw_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor(PURPLE_LIGHT)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Üst mor şerit
    header = Rectangle((0, 0.88), 1, 0.12, facecolor=PURPLE_DARK, edgecolor="none")
    ax.add_patch(header)

    # Sağ üst «ÖNEMLİ» rozeti
    badge_w, badge_h = 0.22, 0.045
    badge_x, badge_y = 1 - badge_w - 0.04, 0.915
    badge = FancyBboxPatch(
        (badge_x, badge_y),
        badge_w,
        badge_h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=WHITE,
        edgecolor=PURPLE_MID,
        linewidth=2,
    )
    ax.add_patch(badge)
    ax.text(
        badge_x + badge_w / 2,
        badge_y + badge_h / 2,
        "ÖNEMLİ",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=PURPLE_DARK,
    )

    ax.text(
        0.05,
        0.935,
        "src/layers.py — Özet",
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=WHITE,
    )
    ax.text(
        0.05,
        0.90,
        "TensorFlow Keras özel attention katmanları",
        ha="left",
        va="center",
        fontsize=10,
        color=PURPLE_ACCENT,
    )

    y = 0.84
    for title, body in SUMMARY_BLOCKS:
        box = FancyBboxPatch(
            (0.06, y - 0.19),
            0.88,
            0.17,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=WHITE,
            edgecolor=PURPLE_MID,
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            0.09,
            y - 0.03,
            title,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=PURPLE_DARK,
        )
        ax.text(
            0.09,
            y - 0.07,
            body,
            ha="left",
            va="top",
            fontsize=9,
            color=TEXT_DARK,
            linespacing=1.35,
        )
        y -= 0.21

    # Alt bilgi
    footer = Rectangle((0, 0), 1, 0.06, facecolor=PURPLE_MID, edgecolor="none")
    ax.add_patch(footer)
    ax.text(
        0.5,
        0.03,
        "Kaynak: src/layers.py  •  ChannelAttention + SpatialAttention",
        ha="center",
        va="center",
        fontsize=8,
        color=WHITE,
    )

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        draw_page(pdf)
    print(f"PDF yazıldı: {OUTPUT}")


if __name__ == "__main__":
    main()
