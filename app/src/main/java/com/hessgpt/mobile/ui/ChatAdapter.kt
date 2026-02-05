package com.hessgpt.mobile.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.hessgpt.mobile.R
import com.hessgpt.mobile.data.Message
import java.text.SimpleDateFormat
import java.util.*

/**
 * Adaptateur pour afficher les messages du chat
 */
class ChatAdapter(
    private val messages: List<Message>
) : RecyclerView.Adapter<ChatAdapter.MessageViewHolder>() {
    
    private val dateFormat = SimpleDateFormat("HH:mm", Locale.getDefault())
    
    companion object {
        private const val VIEW_TYPE_USER = 1
        private const val VIEW_TYPE_ASSISTANT = 2
        private const val VIEW_TYPE_SYSTEM = 3
    }
    
    override fun getItemViewType(position: Int): Int {
        val message = messages[position]
        return when {
            message.isSystem -> VIEW_TYPE_SYSTEM
            message.isUser -> VIEW_TYPE_USER
            else -> VIEW_TYPE_ASSISTANT
        }
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MessageViewHolder {
        val layoutId = when (viewType) {
            VIEW_TYPE_USER -> R.layout.item_message_user
            VIEW_TYPE_ASSISTANT -> R.layout.item_message_assistant
            VIEW_TYPE_SYSTEM -> R.layout.item_message_system
            else -> R.layout.item_message_assistant
        }
        
        val view = LayoutInflater.from(parent.context)
            .inflate(layoutId, parent, false)
        
        return MessageViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: MessageViewHolder, position: Int) {
        val message = messages[position]
        holder.bind(message)
    }
    
    override fun getItemCount(): Int = messages.size
    
    /**
     * ViewHolder pour les messages
     */
    class MessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val messageText: TextView = itemView.findViewById(R.id.messageText)
        private val timeText: TextView? = itemView.findViewById(R.id.timeText)
        
        fun bind(message: Message) {
            messageText.text = message.text
            timeText?.text = SimpleDateFormat("HH:mm", Locale.getDefault())
                .format(Date(message.timestamp))
        }
    }
}
