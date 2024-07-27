?class KrediBasvuruFormu:
    def __init__(self):
        self.ad = input("Adiniz: ")
        self.soyad = input("Soyadiniz: ")
        self.yas = int(input("Yasiniz: "))
        self.gelir = float(input("Aylik Geliriniz: "))
        self.odeme_gecmisi = input("Odeme Gecmisi (iyi/kotu): ")
        self.kredi_kullanim_orani = float(input("Kredi Kullanim Orani: "))
        self.kredi_kullanim_uzunlugu = int(input("Kredi Kullanim Uzunlugu (yil): "))
        self.kredi_turleri = input("Kredi Turleri (virgul ile ayirarak): ")
        self.yeni_kredi_basvurulari = int(input("Yeni Kredi Basvurulari (sayi): "))
        self.toplam_borc = float(input("Toplam Borc: "))

    def kredi_onay_sistemi(self):
        # Basit bir kredi onay sistemi
        if (
            self.yas >= 18 and
            self.gelir > 15000 and
            self.odeme_gecmisi.lower() == "iyi" and
            self.kredi_kullanim_orani < 0.7 and
            self.kredi_kullanim_uzunlugu >= 3 and
            self.yeni_kredi_basvurulari <= 3 and
            self.toplam_borc < self.gelir * 0.33
        ):
            print("Kredi Basvurunuz Onaylandi.")
        else:
            print("uzgunuz, Kredi Basvurunuz Reddedildi.")




kredi_basvuru = KrediBasvuruFormu()

kredi_basvuru.kredi_onay_sistemi()



kredi_basvuru = KrediBasvuruFormu()

kredi_basvuru.kredi_onay_sistemi()

