import tensorflow as tf
"""
variabel adalah sebuah cara mewakiliki status bersama dan persisten yang dimanipulasi
menggunakan tf.Variabel
"""
#untuk membbuat Variabel, berikan nilai awal.tf.variabel akan memuat sebuah dtype yang sama dengan nilai inisialisasi
my_tensor=tf.constant([[1,2],[2,3],[3,4]])
my_variabel=tf.Variable(my_tensor)
print("my tensor ",my_tensor)
print(("my variabel ",my_variabel))#di convert dalam bentuk numpy=array([])
"""
variabel bertindak dan terlihat seperti tensor namun pada kenyataannya variabel tersebut di dukung dengan struktur data tensor, seperti tensor yang memiliki dtype dan dapat di ekspor ke
dalam numpy
"""
print("Shape: ", my_variabel.shape)
print("DType: ", my_variabel.dtype)
print("As NumPy: ", my_variabel.numpy())
#variabel dapat berjalan sesuai operasi pada tensor 
print("A variable:", my_variabel)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variabel))
print("\nIndex of highest value:", tf.argmax(my_variabel))
# This creates a new tensor; it does not reshape the variable.
print(tf.reshape(my_variabel, [1,6]))


#penetapan ulang variabel dapat dilakukan dengan tf.Variable.assign
#=>dengan cara menggunakan memory tensor kembali
a=tf.Variable([2.0,3.0])
print(a.numpy(),a.dtype)#float32
#ini akan menyimpan semua dtype yang sama=>float32
a.assign([1,2])#disini aku masukin integer, ->dtype ttp float 32
print(a.numpy(),a.dtype)
#tidak di izinkan untuk meresizes variabel
try:
    a.assign([1.0,2.0,3.0])
except Exception as e:
    print(f"{type(e).__name__}:{e}")
#=>variabel bereaksi pada backing tensor->membuat variabel yang baru 
# dari variabel yang ada dengan menduplikasi backing tensor,dua variabel
#tidak akan membagi memorynya
a=tf.Variable([2.0, 3.0])
b=tf.Variable(a)
a.assign([5,6])
print(a.numpy())
print(b.numpy())
# There are other versions of assign
print(a.assign_add([2,3]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]


#siklus hidup,penamaan dan menonton
"""
variabel memiliki siklus hidup yang sama dengan object dari pyhton lainnya
dpt memberikan dua variabel nama yang sama

nama variabel sudah ditetapkan secara default, jika ingin menambahkan merupakan
pilihan optional
"""
#MEMBUAT a dan b akan memiliki nama yang sama tapi backing tensor yang berbeda
a=tf.Variable(my_tensor,name='Mark')
# A new variable with the same name, but different value
# Note that the scalar add is broadcast
b=tf.Variable(my_tensor+1,name='Mark')
#mencari kesamaan dari kedua a dan b
print(a==b)
"""
variabel sangat penting terhadap diferensiasi namun kita dpt menonaktifkan gradien
untuk variabel dgn menyetel trainable ->false saat pembuatan
contohnya:
perhitngan langkah pelatihan
"""
step_counter=tf.Variable(1,trainable=False)
print(step_counter)

#menempatkan variabel dan tensor
"""
tensor dan variabel akan ditempatkan pada perangkat tercepatnya yg kompitable dengan
dtypenya, sebgian besar variabel ditempatkan pada gpu jika tersedia
Perhatikan bahwa penempatan perangkat pencatatan harus diaktifkan pada awal sesi.
"""
with tf.device('CPU:0'):
    #create some tensor
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
print(c)
"""
dimungkinkan untuk mengatur tensor dan variabel pada satu perangkat dan 
melakukan perhitungan dengan perangkat lain=>penundaan karena data perlu disalin pada perangkat
->dapat dilakukan jika memiliki banyak pekerja GPU yang hanya menginginkan salinan satu variabel
"""
with tf.device('CPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device('GPU:0'):
  # Element-wise multiply
  k = a * b

print(k)