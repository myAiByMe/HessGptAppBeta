# PyTorch Mobile ProGuard Rules

# Keep PyTorch classes
-keep class org.pytorch.** { *; }
-dontwarn org.pytorch.**

# Keep NativePeer class for PyTorch
-keep class org.pytorch.NativePeer { *; }

# Keep TorchScript serialization classes
-keep class org.pytorch.IValue { *; }
-keep class org.pytorch.Tensor { *; }
-keep class org.pytorch.Module { *; }

# Keep model classes
-keep class com.hessgpt.mobile.ml.** { *; }

# Keep data classes
-keepclassmembers class com.hessgpt.mobile.data.** {
    <fields>;
    <methods>;
}

# Gson
-keepattributes Signature
-keepattributes *Annotation*
-dontwarn sun.misc.**
-keep class com.google.gson.** { *; }
-keep class * implements com.google.gson.TypeAdapterFactory
-keep class * implements com.google.gson.JsonSerializer
-keep class * implements com.google.gson.JsonDeserializer

# Kotlin
-keep class kotlin.** { *; }
-keep class kotlin.Metadata { *; }
-dontwarn kotlin.**
-keepclassmembers class **$WhenMappings {
    <fields>;
}

# Coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}

# Remove logging in release
-assumenosideeffects class android.util.Log {
    public static *** d(...);
    public static *** v(...);
    public static *** i(...);
}

-assumenosideeffects class timber.log.Timber {
    public static *** d(...);
    public static *** v(...);
    public static *** i(...);
}
